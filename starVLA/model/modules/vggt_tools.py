# Copyright 2025 starVLA community. All rights reserved.
# Adapted from ABot-Manipulation's VGGT integration.
"""
VGGT 3D spatial feature extraction and fusion utilities.

- CrossAttention: Residual cross-attention fuser (Q=VLM tokens, K/V=VGGT patches).
- preprocess_images: Resize/crop PIL images to the tensor format expected by VGGT.
- FakeVGGT: Lightweight stub that mimics the real VGGT aggregator API, used for
  smoke-testing when the 1B-parameter checkpoint is not available.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF


# ---------------------------------------------------------------------------
#  CrossAttention fuser  (copied from ABot, untouched)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Residual multi-head cross-attention: fuse spatial KV into query tokens."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        nhead: int = 8,
        dropout: float = 0.0,
        kv_dim: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden if d_hidden is not None else d_model
        self.nhead = nhead
        self.head_dim = self.d_hidden // nhead
        assert self.d_hidden % nhead == 0, "d_hidden must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, self.d_hidden)
        self.k_proj = nn.Linear(kv_dim, self.d_hidden)
        self.v_proj = nn.Linear(kv_dim, self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        image_feature: torch.Tensor,
        spatial_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_feature:   (B, N_img, d_model)   — Query  (VLM hidden states)
            spatial_feature: (B, N_spatial, kv_dim) — Key/Value (VGGT patch tokens)
        Returns:
            fused: (B, N_img, d_model)
        """
        B, N_img, _ = image_feature.shape
        _, N_spatial, _ = spatial_feature.shape

        q = self.q_proj(image_feature)
        k = self.k_proj(spatial_feature)
        v = self.v_proj(spatial_feature)

        q = q.view(B, N_img,     self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_img, self.d_hidden)

        output = self.out_proj(attn_output)
        output = self.dropout_out(output)
        return self.norm(image_feature + output)


# ---------------------------------------------------------------------------
#  Image preprocessing  (copied from ABot, untouched)
# ---------------------------------------------------------------------------

def preprocess_images(image_list, target_size, mode='crop'):
    """Convert a nested list of PIL images to a batched tensor for VGGT.

    Args:
        image_list: List[List[PIL.Image]]  — outer=batch, inner=views.
        target_size: int — target width in pixels.
        mode: 'crop' or 'pad'.

    Returns:
        Tensor of shape (B, V, 3, H, W).
    """
    batch_images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    for imgs in image_list:
        epi_images = []
        for img in imgs:
            width, height = img.size
            if mode == "pad":
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14
            else:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14

            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)

            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y:start_y + target_size, :]

            if mode == "pad":
                h_pad = target_size - img.shape[1]
                w_pad = target_size - img.shape[2]
                if h_pad > 0 or w_pad > 0:
                    pt, pb = h_pad // 2, h_pad - h_pad // 2
                    pl, pr = w_pad // 2, w_pad - w_pad // 2
                    img = F.pad(img, (pl, pr, pt, pb), mode="constant", value=1.0)

            shapes.add((img.shape[1], img.shape[2]))
            epi_images.append(img)
        batch_images.append(torch.stack(epi_images))

    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for img in batch_images:
            hp = max_h - img.shape[2]
            wp = max_w - img.shape[3]
            if hp > 0 or wp > 0:
                pt, pb = hp // 2, hp - hp // 2
                pl, pr = wp // 2, wp - wp // 2
                img = F.pad(img, (pl, pr, pt, pb), mode="constant", value=1.0)
            padded.append(img)
        batch_images = padded

    batch_images = torch.stack(batch_images)
    if len(image_list) == 1 and batch_images.dim() == 3:
        batch_images = batch_images.unsqueeze(0)
    return batch_images


# ---------------------------------------------------------------------------
#  FakeVGGT — test stub (same API as the real VGGT aggregator)
# ---------------------------------------------------------------------------

class _FakeAggregator(nn.Module):
    """Mimics ``VGGT.aggregator(images)`` for smoke-testing."""

    def __init__(self, token_dim: int = 2048, patch_size: int = 14,
                 num_registers: int = 4):
        super().__init__()
        self.token_dim = token_dim
        self.patch_size = patch_size
        self.num_registers = num_registers  # CLS + register tokens before patches

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, V, 3, H, W)
        Returns:
            aggregated_tokens_list: List with one element of shape
                                    (B, V, num_registers + N_patch, token_dim)
            ps_idx: int — start index of patch tokens.
        """
        B, V, C, H, W = images.shape
        N_patch = (H // self.patch_size) * (W // self.patch_size)
        total = self.num_registers + N_patch
        fake_tokens = torch.randn(
            B, V, total, self.token_dim,
            device=images.device, dtype=images.dtype,
        )
        return [fake_tokens], self.num_registers


class FakeVGGT(nn.Module):
    """Drop-in replacement for ``VGGT.from_pretrained(...)`` during testing."""

    def __init__(self):
        super().__init__()
        self.aggregator = _FakeAggregator()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
