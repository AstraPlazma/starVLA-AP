# Copyright 2025 starVLA community. All rights reserved.
# QwenContr Framework
# Based on QwenGR00T with the following enhancement:
#   - Uses ContrFlowmatchingActionHead (Heun ODE solver for inference)

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.Contr_ActionHeader import get_contr_action_model, ContrFlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenContr")
class Qwen_Contr(baseframework):
    """
    QwenContr: Qwen VLM + ContrFlowmatchingActionHead (Heun ODE solver).

    Identical to QwenGR00T in architecture and training, but uses Heun's
    2nd-order method during inference for improved action prediction accuracy.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Align cross-attention dim with VLM hidden size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: ContrFlowmatchingActionHead = get_contr_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Step 1: QwenVL encoding
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        # Step 2: Action head forward and loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            # Build task labels from instruction strings for contrastive sampling
            task_labels = torch.tensor(
                [hash(inst) % (2**31) for inst in instructions],
                device=last_hidden.device, dtype=torch.long,
            )
            task_labels_repeated = task_labels.repeat(repeated_diffusion_steps)

            action_output = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                task_labels=task_labels_repeated,
            )

        return {
            "action_loss": action_output["loss"],
            "pos_error": action_output["pos_error"],
            "neg_error": action_output["neg_error"],
            "contrastive_active": action_output["contrastive_active"],
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None

        # Inference with Heun solver (use_heun=True by default in ContrFlowmatchingActionHead)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "QwenContr"

    model: Qwen_Contr = Qwen_Contr(cfg)
    print(model)

    # Fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Pick up the red block and place it on the blue target.",
    }

    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Predicted Action shape: {normalized_actions.shape}")
    print("QwenContr smoke test passed.")
