# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n starVLA python=3.10 -y && conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

Verified: `flash-attn==2.7.4.post1` works with CUDA 12.0/12.4.

## Commands

```bash
# Lint/format check (CI gate)
make check          # black + ruff, read-only
make autoformat     # black + ruff, in-place

# Smoke test a framework (requires model in playground/Pretrained_models/)
python starVLA/model/framework/QwenGR00T.py

# Smoke test dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml

# Training (single/multi-GPU via Accelerate + DeepSpeed)
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml

# Evaluation (LIBERO example)
bash examples/LIBERO/eval_files/run_policy_server.sh &
bash examples/LIBERO/eval_files/eval_libero.sh
```

## Architecture

StarVLA is a **modular VLA (Vision-Language-Action) framework** following high-cohesion/low-coupling design. Each component is independently debuggable.

### Data Flow

```
Raw inputs (PIL images, lang str, action np.ndarray)
  └─> DataLoader (model-agnostic dict, no preprocessing)
        └─> Framework.forward() / Framework.predict_action()
              ├─> VLM module (Qwen2.5-VL / Qwen3-VL / Florence-2)
              ├─> [Optional] DINO encoder (dense spatial tokens)
              └─> Action Head (FM/DiT/MLP/Fast-tokenizer)
```

### Key Directories

- `starVLA/model/framework/` — **Top-level model assemblies** (the primary API surface). Each `*.py` can run standalone. Registered via `@FRAMEWORK_REGISTRY.register("Name")`. Base class: `baseframework(PreTrainedModel)` in `base_framework.py`.
- `starVLA/model/modules/` — Pluggable sub-components:
  - `vlm/` — VLM wrappers: `QWen2_5.py`, `QWen3_5.py`, `Florence2.py`, etc.
  - `action_model/` — Action heads: `GR00T_ActionHeader.py` (flow-matching DiT), `MLP_ActionHeader.py`, `fast_ActionHeader.py`, etc.
  - `memory/` — Memory modules
  - `projector/`, `dino_model/`
- `starVLA/training/` — Training scripts:
  - `train_starvla.py` — Standard VLA training (PyTorch + Accelerate + DeepSpeed)
  - `train_starvla_cotrain.py` — Co-training VLA + VLM multimodal data
  - `train_starvlm.py` — VLM-only training
- `starVLA/config/training/` — YAML configs (OmegaConf). Primary entry: `starvla_cotrain_libero.yaml` and `starvla_cotrain_oxe.yaml`
- `starVLA/dataloader/` — `lerobot_datasets.py` is the VLA dataloader (LeRobot format)
- `deployment/` — WebSocket model server for real-robot inference
- `examples/` — Benchmark-specific train/eval scripts (LIBERO, RoboCasa, Calvin, SimplerEnv, etc.)
- `playground/` — Local model checkpoints and datasets (not packaged)

### Supported VLA Frameworks

| Name | Description |
|------|-------------|
| `QwenGR00T` | Qwen2.5-VL + flow-matching DiT head (dual-system, like GR00T) |
| `QwenPI` | Qwen + flow-matching expert (like π₀) |
| `QwenOFT` | Qwen + MLP head, parallel decode (like OpenVLA-OFT) |
| `QwenFast` | Qwen + fast tokenizer, autoregressive discrete actions (like π₀-fast) |

### Configuration System

Config is loaded via `OmegaConf.load(args.config_yaml)`. CLI overrides use dot-notation:
```bash
--framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct
--trainer.freeze_modules "qwen_vl_interface.model.model.visual"
```

Per-module learning rates are set in YAML:
```yaml
trainer:
  learning_rate:
    base: 1e-5
    action_model: 1e-4
```

Checkpoint resumption: `trainer.pretrained_checkpoint` + `trainer.reload_modules` (empty = load all). Note: optimizer state is **not** saved.

### Pretrained Models Location

Place pretrained models under `./playground/Pretrained_models/`. For quick check, download [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) to `./playground/Pretrained_models/Qwen3-VL-4B-Instruct`.
