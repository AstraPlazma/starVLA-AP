# Copyright 2025 starVLA community. All rights reserved.
# Contr ActionHeader: FlowmatchingActionHead with Heun ODE solver + Triplet Contrastive Loss
# Based on GR00T_ActionHeader.py with the following enhancements:
#   - Heun's method (2nd-order) for predict_action, improving accuracy per step
#   - Class-conditioned triplet contrastive loss (from DeltaFM/triplet_loss.py)

import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class ContrFlowmatchingActionHead(nn.Module):
    """Flow-matching action head with Heun's method for inference.

    Identical to FlowmatchingActionHead in architecture and training,
    but uses Heun's 2nd-order ODE solver during predict_action for
    improved accuracy at the same number of discretization steps.
    """

    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]

        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

        # Contrastive loss config (0 = disabled, pure MSE)
        self.contrastive_weight = float(getattr(config, "contrastive_weight", 0.0))

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    # ------------------------------------------------------------------
    # Contrastive loss (adapted from DeltaFM/triplet_loss.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _class_conditioned_sampling(labels):
        """Sample one negative index per sample, ensuring different task label.

        Args:
            labels: (B,) integer task labels.

        Returns:
            (B,) negative indices, or None if all labels are identical.
        """
        bsz = labels.shape[0]
        mask = ~(labels[None] == labels[:, None])  # (B, B), True where labels differ
        weights = mask.float()
        weights_sum = weights.sum(dim=1, keepdim=True)

        if (weights_sum == 0).all():
            return None  # all same task — no valid negatives

        if (weights_sum == 0).any():
            # Some samples have no valid negatives (all same label in their row).
            # Fallback: uniform random for those rows (will be masked later if needed).
            fallback = torch.randint(0, bsz, (bsz,), device=labels.device)
            valid_mask = (weights_sum.squeeze(1) > 0)
            weights[~valid_mask] = 1.0
            weights[~valid_mask] = weights[~valid_mask] / weights[~valid_mask].sum(dim=1, keepdim=True)

        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        choices = torch.multinomial(weights, 1).squeeze(1)
        return choices

    def _compute_contrastive_loss(self, pred, velocity, task_labels):
        """Triplet contrastive loss with class-conditioned negative sampling.

        Args:
            pred: (B, T, action_dim) predicted velocity.
            velocity: (B, T, action_dim) target velocity.
            task_labels: (B,) integer task labels.

        Returns:
            dict: {loss, pos_error, neg_error, contrastive_active}
        """
        pred_flat = pred.flatten(1)          # (B, T*action_dim)
        velocity_flat = velocity.flatten(1)  # (B, T*action_dim)

        # Positive error: standard MSE
        pos_error = ((pred_flat - velocity_flat) ** 2).mean(dim=1)  # (B,)

        # Negative sampling: pick a sample from a different task
        neg_indices = self._class_conditioned_sampling(task_labels)

        if neg_indices is None:
            # All samples in batch share the same task — no valid negatives.
            warnings.warn(
                "[ContrActionHeader] All samples in batch share the same task label. "
                "Contrastive loss disabled for this step (falling back to pure MSE).",
                stacklevel=2,
            )
            mse = pos_error.mean()
            return {"loss": mse, "pos_error": mse, "neg_error": mse.new_tensor(0.0), "contrastive_active": False}

        velocity_neg = velocity_flat[neg_indices]  # (B, T*action_dim)
        neg_error = ((pred_flat - velocity_neg) ** 2).mean(dim=1)  # (B,)

        loss = pos_error - self.contrastive_weight * neg_error  # (B,)
        return {
            "loss": loss.mean(),
            "pos_error": pos_error.mean(),
            "neg_error": neg_error.mean(),
            "contrastive_active": True,
        }

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None,
                encoder_attention_mask=None, task_labels=None):
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        # Loss: contrastive triplet (if enabled and labels provided) or standard MSE
        if task_labels is not None and self.contrastive_weight > 0:
            loss_dict = self._compute_contrastive_loss(pred_actions, velocity, task_labels)
        else:
            mse = ((pred_actions - velocity) ** 2).mean()
            loss_dict = {"loss": mse, "pos_error": mse, "neg_error": mse.new_tensor(0.0), "contrastive_active": False}

        return loss_dict

    # ------------------------------------------------------------------
    # Inference with Heun's method
    # ------------------------------------------------------------------

    def _predict_velocity(self, actions, t_discretized, vl_embs, state_features):
        """Shared forward pass: encode actions -> DiT -> decode velocity."""
        batch_size = vl_embs.shape[0]
        device = vl_embs.device

        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        action_features = self.action_encoder(actions, timesteps_tensor)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred = self.action_decoder(model_output)
        return pred[:, -self.action_horizon:]

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, use_heun: bool = True) -> torch.Tensor:
        """Predict action trajectory from noise using flow matching ODE.

        Args:
            vl_embs: VLM hidden states (B, L, H).
            state: Optional proprioceptive state (B, 1, state_dim).
            use_heun: If True, use Heun's 2nd-order method; otherwise Euler.

        Returns:
            Predicted actions (B, action_horizon, action_dim).
        """
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            v1 = self._predict_velocity(actions, t_discretized, vl_embs, state_features)

            if use_heun and t < num_steps - 1:
                # Heun's method: evaluate velocity at the Euler-predicted next point,
                # then average the two velocities for a 2nd-order update.
                actions_euler = actions + dt * v1
                t_next_cont = (t + 1) / float(num_steps)
                t_next_discretized = int(t_next_cont * self.num_timestep_buckets)
                v2 = self._predict_velocity(actions_euler, t_next_discretized, vl_embs, state_features)
                actions = actions + dt * 0.5 * (v1 + v2)
            else:
                # Last step or Heun disabled: plain Euler
                actions = actions + dt * v1

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_contr_action_model(config=None):
    """Factory: build ContrFlowmatchingActionHead from global framework config."""
    return ContrFlowmatchingActionHead(full_config=config)
