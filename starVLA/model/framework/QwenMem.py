# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
QwenMem Framework
Qwen-VL with dual memory mechanism (Cognitive + Perceptual) + Flow-matching action head
Memory mechanism adapted from MemoryVLA, Flow-matching header from GR00T N1.5
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.memory import BottleneckSE, CogMemBank, PerMemBank


@FRAMEWORK_REGISTRY.register("QwenMem")
class QwenMem(baseframework):
    """
    Multimodal vision-language-action model with dual memory mechanism.

    Components:
      - Qwen3.5 VL interface for fused language/vision token embeddings
      - Dual memory banks: CogMemBank (cognitive) + PerMemBank (perceptual)
      - BottleneckSE for perception feature compression
      - DiT diffusion head with perception attention for action prediction

    Focus: Predict future continuous actions with episodic memory enhancement.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Memory mechanism parameters
        self.cog_token_size = self.qwen_vl_interface.model.config.hidden_size

        # Get vision_dim from Qwen config (not lazy initialization)
        self.vision_dim = self.qwen_vl_interface.model.config.vision_config.hidden_size

        # Get memory config with defaults
        mem_cfg = config.framework.get("memory", {})
        self.per_token_size = mem_cfg.get("per_token_size", 256)
        self.mem_length = mem_cfg.get("mem_length", 16)
        self.retrieval_layers = mem_cfg.get("retrieval_layers", 2)
        self.use_timestep_pe = mem_cfg.get("use_timestep_pe", True)
        self.fusion_type = mem_cfg.get("fusion_type", "gate")
        self.consolidate_type = mem_cfg.get("consolidate_type", "tome")
        self.update_fused = mem_cfg.get("update_fused", False)

        # Get dataloader config
        data_cfg = config.datasets.get("vla_data", {})
        self.dataloader_type = data_cfg.get("dataloader_type", "group")
        self.group_size = data_cfg.get("group_size", 16)

        # Initialize memory modules immediately (for optimizer to see parameters)
        self.per_compr = BottleneckSE(
            C_in=self.vision_dim,
            C_mid=self.per_token_size * 2,
            C_out=self.per_token_size,
        )

        self.cog_mem_bank = CogMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.cog_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
        )

        self.per_mem_bank = PerMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.per_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
        )

        # Inference state tracking
        self.cur_timestep = 0
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Extract episode_ids and timesteps for memory mechanism
        episode_ids = np.array([example.get("episode_id", 0) for example in examples])
        timesteps = np.array([example.get("timestep", 0) for example in examples])

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

            # Extract vision features for perception memory.
            # Detach to stop gradients from flowing back through the visual encoder
            # via the per_tokens path (matches MemoryVLA where VLM is frozen).
            # The visual encoder still receives gradients via the cog_tokens → last_hidden path.
            vision_feats = self.qwen_vl_interface.vision_feats.detach()  # [B, N, D]

            # Extract spatial shape from image_grid_thw
            image_grid_thw = qwen_inputs.get('image_grid_thw', None)  # [B, 3] where 3=[t,h,w]

            # Set vision_dim on first forward pass
            if self.vision_dim is None:
                self.vision_dim = vision_feats.shape[-1]
                logger.info(f"Vision dimension detected: {self.vision_dim}")

                # Initialize memory modules
                self.per_compr = BottleneckSE(
                    C_in=self.vision_dim,
                    C_mid=self.per_token_size * 2,
                    C_out=self.per_token_size,
                ).to(vision_feats.device)

                self.cog_mem_bank = CogMemBank(
                    dataloader_type=self.dataloader_type,
                    group_size=self.group_size,
                    token_size=self.cog_token_size,
                    mem_length=self.mem_length,
                    retrieval_layers=self.retrieval_layers,
                    use_timestep_pe=self.use_timestep_pe,
                    fusion_type=self.fusion_type,
                    consolidate_type=self.consolidate_type,
                    update_fused=self.update_fused,
                ).to(vision_feats.device)

                self.per_mem_bank = PerMemBank(
                    dataloader_type=self.dataloader_type,
                    group_size=self.group_size,
                    token_size=self.per_token_size,
                    mem_length=self.mem_length,
                    retrieval_layers=self.retrieval_layers,
                    use_timestep_pe=self.use_timestep_pe,
                    fusion_type=self.fusion_type,
                    consolidate_type=self.consolidate_type,
                    update_fused=self.update_fused,
                ).to(vision_feats.device)

                logger.info("Memory modules initialized")

            # Process with dual memory mechanism (if modules initialized)
            if self.cog_mem_bank is not None and self.per_mem_bank is not None:

                # Extract cognition tokens (use attention_mask to find last valid token)
                attention_mask = qwen_inputs['attention_mask']
                cumulative_sum = attention_mask.cumsum(dim=1)
                last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
                expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
                cog_tokens = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, H]

                # Compress perception features with spatial shape
                if image_grid_thw is not None:
                    # Extract h, w from first image's grid_thw
                    spatial_h = int(image_grid_thw[0, 1].item())
                    spatial_w = int(image_grid_thw[0, 2].item())
                    per_tokens = self.per_compr(vision_feats, spatial_shape=(spatial_h, spatial_w))
                else:
                    per_tokens = self.per_compr(vision_feats)

                # Process through memory banks
                cog_tokens = self.cog_mem_bank.process_batch(
                    tokens=cog_tokens,
                    episode_ids=episode_ids,
                    timesteps=timesteps,
                )  # [B, 1, H]

                per_tokens = self.per_mem_bank.process_batch(
                    tokens=per_tokens,
                    episode_ids=episode_ids,
                    timesteps=timesteps,
                )  # [B, N, per_token_size]
            else:
                per_tokens = None

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=cog_tokens.device, dtype=cog_tokens.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            cog_tokens_repeated = cog_tokens.repeat(repeated_diffusion_steps, 1, 1)  # [4B, 1, H]
            per_tokens_repeated = per_tokens.repeat(repeated_diffusion_steps, 1, 1) if per_tokens is not None else None

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=cog_tokens.device, dtype=cog_tokens.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(cog_tokens_repeated, actions_target_repeated, state_repeated, per_tokens=per_tokens_repeated)



        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        episode_first_frame: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Predict action with memory mechanism.

        Args:
            examples: List of input examples
            episode_first_frame: If True, reset memory banks for new episode
            **kwargs: Additional arguments

        Returns:
            dict: normalized_actions (np.ndarray): Shape [B, T, action_dim]
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # QWenVL forward
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            vision_feats = self.qwen_vl_interface.vision_feats

            # Extract spatial shape from image_grid_thw
            image_grid_thw = qwen_inputs.get('image_grid_thw', None)

            per_tokens = None

            # Process with memory mechanism
            if self.cog_mem_bank is not None and self.per_mem_bank is not None:
                # Reset memory if new episode
                if episode_first_frame:
                    self.cog_mem_bank.reset()
                    self.per_mem_bank.reset()
                    self.cur_timestep = 0

                # Extract cognition tokens
                attention_mask = qwen_inputs['attention_mask']
                cumulative_sum = attention_mask.cumsum(dim=1)
                last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
                expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
                cog_tokens = last_hidden.gather(1, expanded_indices.unsqueeze(1))

                # Compress perception features
                if image_grid_thw is not None:
                    spatial_shape = (image_grid_thw[0, 1].item(), image_grid_thw[0, 2].item())
                    per_tokens = self.per_compr(vision_feats, spatial_shape)
                else:
                    per_tokens = self.per_compr(vision_feats)

                # Process through memory banks
                episode_ids = np.array([0] * len(examples))
                timesteps = np.array([self.cur_timestep] * len(examples))
                self.cur_timestep += 1

                cog_tokens = self.cog_mem_bank.process_batch(
                    tokens=cog_tokens,
                    episode_ids=episode_ids,
                    timesteps=timesteps,
                )

                per_tokens = self.per_mem_bank.process_batch(
                    tokens=per_tokens,
                    episode_ids=episode_ids,
                    timesteps=timesteps,
                )

        state = torch.from_numpy(np.array(state)).to(cog_tokens.device, dtype=cog_tokens.dtype) if state is not None else None

        # Action prediction with dual memory: cog_tokens as sole conditioning (same as MemoryVLA)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(cog_tokens, state, per_tokens=per_tokens)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    model: QwenMem = QwenMem(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # forward model with dataloader
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)
    print("Finished")
