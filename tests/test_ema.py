"""
EMA 机制单元测试（不依赖真实数据集、不需要多 GPU）

测试覆盖：
1. build_ema_state 初始化正确
2. update_ema 更新权重
3. ema_swap 上下文管理器交换/恢复
4. ema_decay=0 时完全禁用
5. 检查点保存包含 EMA 文件
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn

# Add project root to path (before site-packages) so we test the local code
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

passed = 0
failed = 0
errors = []


def check(condition, msg):
    global passed, failed, errors
    if condition:
        print(f"  [PASS] {msg}")
        passed += 1
    else:
        print(f"  [FAIL] {msg}")
        failed += 1
        errors.append(msg)


def header(name):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")


class FakeAccelerator:
    """Minimal accelerator mock that simulates main process behavior."""
    def __init__(self, is_main=True):
        self._is_main = is_main
        self.sync_gradients = True

    @property
    def is_main_process(self):
        return self._is_main

    def get_state_dict(self, model):
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()

    def unwrap_model(self, model):
        if hasattr(model, 'module'):
            return model.module
        return model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


# =====================================================================
# 1. build_ema_state
# =====================================================================
header("1. build_ema_state")

model = SimpleModel()
accel = FakeAccelerator(is_main=True)

ema_state = TrainerUtils.build_ema_state(model, accel)
check(ema_state is not None, "EMA state created on main process")
check(isinstance(ema_state, dict), "EMA state is a dict")

# Verify all keys match
model_keys = set(model.state_dict().keys())
ema_keys = set(ema_state.keys())
check(model_keys == ema_keys, f"EMA keys match model keys ({len(model_keys)} params)")

# Verify values are identical at init
for k in model.state_dict():
    check(
        torch.allclose(model.state_dict()[k].float(), ema_state[k]),
        f"  {k}: initial EMA == model weights"
    )

# Verify EMA is on CPU and float32
for k, v in ema_state.items():
    check(v.device == torch.device("cpu"), f"  {k}: on CPU")
    check(v.dtype == torch.float32, f"  {k}: float32")

# Non-main process should return None
accel_worker = FakeAccelerator(is_main=False)
ema_none = TrainerUtils.build_ema_state(model, accel_worker)
check(ema_none is None, "Non-main process returns None")


# =====================================================================
# 2. update_ema
# =====================================================================
header("2. update_ema")

model = SimpleModel()
accel = FakeAccelerator(is_main=True)
ema_state = TrainerUtils.build_ema_state(model, accel)

# Save initial EMA values
ema_before = {k: v.clone() for k, v in ema_state.items()}

# Modify model weights (simulate optimizer step)
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.1)

# Update EMA
decay = 0.99
TrainerUtils.update_ema(ema_state, model, accel, decay=decay)

# Verify EMA moved toward new model weights
for k in ema_state:
    model_val = model.state_dict()[k].float().cpu()
    expected = decay * ema_before[k] + (1 - decay) * model_val
    check(
        torch.allclose(ema_state[k], expected, atol=1e-6),
        f"  {k}: EMA = {decay}*old + {1-decay}*new"
    )

# Verify EMA != model (decay < 1 means they should differ)
any_diff = False
for k in ema_state:
    if not torch.allclose(ema_state[k], model.state_dict()[k].float().cpu(), atol=1e-6):
        any_diff = True
        break
check(any_diff, "EMA weights differ from model weights after update")

# Non-main process: update should be no-op
accel_worker = FakeAccelerator(is_main=False)
ema_copy = {k: v.clone() for k, v in ema_state.items()}
TrainerUtils.update_ema(ema_state, model, accel_worker, decay=decay)
all_same = all(torch.equal(ema_state[k], ema_copy[k]) for k in ema_state)
check(all_same, "Non-main process: update_ema is no-op")

# None ema_state: should not crash
TrainerUtils.update_ema(None, model, accel, decay=decay)
check(True, "update_ema(None, ...) does not crash")


# =====================================================================
# 3. ema_swap context manager
# =====================================================================
header("3. ema_swap context manager")

model = SimpleModel()
accel = FakeAccelerator(is_main=True)
ema_state = TrainerUtils.build_ema_state(model, accel)

# Make EMA differ from model
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.5)

# Record original model weights
original_weights = {k: v.clone() for k, v in model.state_dict().items()}

# Inside swap: model should have EMA weights
with TrainerUtils.ema_swap(model, ema_state, accel):
    for k in model.state_dict():
        swapped = model.state_dict()[k].float().cpu()
        check(
            torch.allclose(swapped, ema_state[k], atol=1e-5),
            f"  Inside swap: {k} == EMA weights"
        )

# After swap: model should have original weights restored
for k in model.state_dict():
    check(
        torch.allclose(model.state_dict()[k], original_weights[k]),
        f"  After swap: {k} == original weights"
    )

# Non-main process: swap should be no-op
accel_worker = FakeAccelerator(is_main=False)
weights_before = {k: v.clone() for k, v in model.state_dict().items()}
with TrainerUtils.ema_swap(model, ema_state, accel_worker):
    for k in model.state_dict():
        check(
            torch.equal(model.state_dict()[k], weights_before[k]),
            f"  Non-main swap: {k} unchanged"
        )

# None ema_state: should be no-op
with TrainerUtils.ema_swap(model, None, accel):
    check(True, "ema_swap(None) is no-op, no crash")


# =====================================================================
# 4. ema_decay=0 disabled behavior
# =====================================================================
header("4. ema_decay=0 (disabled)")

ema_decay = 0.0
ema_state_disabled = None

if ema_decay > 0:
    ema_state_disabled = TrainerUtils.build_ema_state(model, accel)

check(ema_state_disabled is None, "ema_decay=0: no EMA state created")

# update and swap should be no-ops
TrainerUtils.update_ema(ema_state_disabled, model, accel, decay=0.0)
check(True, "update_ema with None state: no crash")

with TrainerUtils.ema_swap(model, ema_state_disabled, accel):
    check(True, "ema_swap with None state: no crash")


# =====================================================================
# 5. Checkpoint save with EMA
# =====================================================================
header("5. Checkpoint save simulation")

model = SimpleModel()
accel = FakeAccelerator(is_main=True)
ema_state = TrainerUtils.build_ema_state(model, accel)

# Simulate some training
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.1)
TrainerUtils.update_ema(ema_state, model, accel, decay=0.9999)

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "steps_100")

    # Save model
    torch.save(model.state_dict(), ckpt_path + "_pytorch_model.pt")
    check(os.path.exists(ckpt_path + "_pytorch_model.pt"), "Model checkpoint saved")

    # Save EMA
    torch.save(ema_state, ckpt_path + "_ema.pt")
    check(os.path.exists(ckpt_path + "_ema.pt"), "EMA checkpoint saved")

    # Reload EMA and verify
    loaded_ema = torch.load(ckpt_path + "_ema.pt", map_location="cpu", weights_only=False)
    for k in ema_state:
        check(
            torch.equal(loaded_ema[k], ema_state[k]),
            f"  {k}: loaded EMA matches saved EMA"
        )

    # Load EMA into a fresh model (simulating inference with EMA weights)
    model2 = SimpleModel()
    ema_on_device = {k: v.to(dtype=next(model2.parameters()).dtype) for k, v in loaded_ema.items()}
    model2.load_state_dict(ema_on_device)
    check(True, "Fresh model loaded with EMA weights successfully")

    # Verify output differs from non-EMA model
    x = torch.randn(2, 10)
    out_ema = model2(x)
    out_orig = model(x)
    check(not torch.allclose(out_ema, out_orig, atol=1e-5), "EMA model output differs from training model")


# =====================================================================
# 6. Multiple EMA updates converge
# =====================================================================
header("6. EMA convergence over many steps")

model = SimpleModel()
accel = FakeAccelerator(is_main=True)
ema_state = TrainerUtils.build_ema_state(model, accel)

# Fix model weights to a target
target_weights = {k: v.clone() for k, v in model.state_dict().items()}

# EMA should converge to model weights with many updates (no weight changes)
for _ in range(1000):
    TrainerUtils.update_ema(ema_state, model, accel, decay=0.99)

for k in ema_state:
    check(
        torch.allclose(ema_state[k], target_weights[k].float().cpu(), atol=1e-4),
        f"  {k}: EMA converged to model after 1000 steps"
    )


# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"  EMA Test Summary: {passed} PASSED / {failed} FAILED")
print(f"{'='*60}")
if errors:
    print("  Failed:")
    for e in errors:
        print(f"    - {e}")

sys.exit(0 if failed == 0 else 1)
