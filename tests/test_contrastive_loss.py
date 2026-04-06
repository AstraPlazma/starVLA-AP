"""
Contrastive Loss 单元测试
验证 Contr_ActionHeader 的 triplet contrastive loss 功能。
"""

import os
import sys
import warnings

import torch
from omegaconf import OmegaConf

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

from starVLA.model.modules.action_model.Contr_ActionHeader import ContrFlowmatchingActionHead

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


def make_config(contrastive_weight=0.05):
    return OmegaConf.create({
        'framework': {
            'action_model': {
                'hidden_size': 256,
                'action_model_type': 'DiT-B',
                'action_dim': 7,
                'future_action_window_size': 15,
                'num_inference_timesteps': 10,
                'state_dim': 0,
                'num_target_vision_tokens': 32,
                'add_pos_embed': True,
                'max_seq_len': 512,
                'noise_beta_alpha': 1.5,
                'noise_beta_beta': 1.0,
                'noise_s': 0.999,
                'num_timestep_buckets': 1000,
                'contrastive_weight': contrastive_weight,
                'diffusion_model_cfg': {
                    'num_layers': 2,
                    'output_dim': 64,
                    'cross_attention_dim': 128,
                }
            }
        }
    })


B = 4
T = 16
D_action = 7
D_vl = 128
L_vl = 20

# =====================================================================
# 1. contrastive_weight=0 时行为与纯 MSE 一致
# =====================================================================
header("1. contrastive_weight=0 -> pure MSE")

head_mse = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.0))
check(head_mse.contrastive_weight == 0.0, "contrastive_weight=0 read from config")

vl = torch.randn(B, L_vl, D_vl)
actions = torch.randn(B, T, D_action)
labels = torch.tensor([0, 1, 2, 3])

torch.manual_seed(42)
out_no_label = head_mse(vl, actions)
torch.manual_seed(42)
out_with_label = head_mse(vl, actions, task_labels=labels)

check(
    abs(out_no_label["loss"].item() - out_with_label["loss"].item()) < 1e-6,
    f"weight=0: loss identical with/without labels ({out_no_label['loss'].item():.6f} == {out_with_label['loss'].item():.6f})"
)

# =====================================================================
# 2. 不同 label 时，contrastive loss 生效
# =====================================================================
header("2. Different labels -> contrastive loss active")

head = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.05))
check(head.contrastive_weight == 0.05, "contrastive_weight=0.05 read from config")

# All different labels
labels_diff = torch.tensor([0, 1, 2, 3])
torch.manual_seed(42)
out_contr = head(vl, actions, task_labels=labels_diff)

torch.manual_seed(42)
out_pure = head_mse(vl, actions)

check(torch.isfinite(out_contr["loss"]), f"contrastive loss finite: {out_contr['loss'].item():.6f}")
check(out_contr["contrastive_active"] == True, "contrastive_active=True")
check(out_contr["neg_error"].item() > 0, f"neg_error > 0: {out_contr['neg_error'].item():.6f}")
check(
    abs(out_contr["loss"].item() - out_pure["loss"].item()) > 1e-4,
    f"contrastive loss differs from MSE ({out_contr['loss'].item():.6f} vs {out_pure['loss'].item():.6f})"
)

# =====================================================================
# 3. 全同 label 退化为纯 MSE + 警告
# =====================================================================
header("3. All same labels -> fallback to MSE + warning")

labels_same = torch.tensor([5, 5, 5, 5])

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    torch.manual_seed(42)
    out_same = head(vl, actions, task_labels=labels_same)

    warning_fired = any("same task label" in str(warning.message) for warning in w)
    check(warning_fired, "Warning printed for all-same-label batch")

check(out_same["contrastive_active"] == False, "contrastive_active=False for all-same")
check(out_same["neg_error"].item() == 0.0, "neg_error=0 for all-same")

# Compare against same model with task_labels=None (should be identical = pure MSE path)
torch.manual_seed(42)
out_no_contr = head(vl, actions, task_labels=None)

check(
    abs(out_same["loss"].item() - out_no_contr["loss"].item()) < 1e-5,
    f"all-same fallback equals no-label path ({out_same['loss'].item():.6f} == {out_no_contr['loss'].item():.6f})"
)

# =====================================================================
# 4. repeated_diffusion_steps 下 label 正确传播
# =====================================================================
header("4. Repeated diffusion steps label propagation")

repeated = 4
labels_base = torch.tensor([10, 20, 30, 40])
labels_repeated = labels_base.repeat(repeated)
check(labels_repeated.shape[0] == B * repeated, f"repeated labels shape: {labels_repeated.shape[0]}")

vl_rep = vl.repeat(repeated, 1, 1)
actions_rep = actions.repeat(repeated, 1, 1)

out_rep = head(vl_rep, actions_rep, task_labels=labels_repeated)
check(torch.isfinite(out_rep["loss"]), f"repeated batch loss finite: {out_rep['loss'].item():.6f}")
check(out_rep["contrastive_active"] == True, "repeated batch: contrastive_active=True")

# =====================================================================
# 5. 反向传播正常
# =====================================================================
header("5. Backward pass")

head_bp = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.05))
vl_bp = torch.randn(B, L_vl, D_vl)
act_bp = torch.randn(B, T, D_action)
labels_bp = torch.tensor([0, 1, 0, 2])

out_bp = head_bp(vl_bp, act_bp, task_labels=labels_bp)
out_bp["loss"].backward()

has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_bp.parameters())
check(has_grad, "Gradients flow through contrastive loss")

has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in head_bp.parameters())
check(not has_nan, "No NaN gradients")

# =====================================================================
# 6. _class_conditioned_sampling 正确性
# =====================================================================
header("6. _class_conditioned_sampling correctness")

# All different
labels_test = torch.tensor([0, 1, 2, 3])
indices = ContrFlowmatchingActionHead._class_conditioned_sampling(labels_test)
check(indices is not None, "all-different: returns indices")
for i in range(4):
    check(labels_test[indices[i]] != labels_test[i], f"  sample {i}: negative has different label")

# Some same
labels_mixed = torch.tensor([0, 0, 1, 1])
indices_m = ContrFlowmatchingActionHead._class_conditioned_sampling(labels_mixed)
check(indices_m is not None, "mixed: returns indices")
for i in range(4):
    check(labels_mixed[indices_m[i]] != labels_mixed[i],
          f"  sample {i} (label={labels_mixed[i].item()}): negative label={labels_mixed[indices_m[i]].item()}")

# All same
labels_all = torch.tensor([7, 7, 7, 7])
indices_all = ContrFlowmatchingActionHead._class_conditioned_sampling(labels_all)
check(indices_all is None, "all-same: returns None")

# =====================================================================
# 7. task_labels=None 不触发对比（即使 weight>0）
# =====================================================================
header("7. task_labels=None -> no contrastive")

# Use the SAME model with weight=0.05, compare with/without labels=None
# Both should take the MSE path when labels=None
head_w = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.1))

torch.manual_seed(42)
out_none = head_w(vl, actions, task_labels=None)

check(torch.isfinite(out_none["loss"]), f"task_labels=None: loss finite ({out_none['loss'].item():.6f})")
check(out_none["contrastive_active"] == False, "task_labels=None: contrastive_active=False")

# Verify it matches same model called again with same seed (reproducibility, i.e. it's deterministic MSE)
torch.manual_seed(42)
out_none2 = head_w(vl, actions, task_labels=None)
check(
    abs(out_none["loss"].item() - out_none2["loss"].item()) < 1e-6,
    f"task_labels=None: deterministic MSE ({out_none['loss'].item():.6f} == {out_none2['loss'].item():.6f})"
)

# =====================================================================
# 8. Hash-based labels 模拟 QwenContr.forward() 的做法
# =====================================================================
header("8. Hash-based labels from instruction strings")

instructions = [
    "Pick up the red block",
    "Place the cup on the shelf",
    "Pick up the red block",  # same as [0]
    "Open the drawer",
]

task_labels_hash = torch.tensor([hash(inst) % (2**31) for inst in instructions], dtype=torch.long)
check(task_labels_hash[0] == task_labels_hash[2], "same instruction -> same label hash")
check(task_labels_hash[0] != task_labels_hash[1], "different instruction -> different hash")
check(task_labels_hash[1] != task_labels_hash[3], "different instruction -> different hash (2)")

indices_hash = ContrFlowmatchingActionHead._class_conditioned_sampling(task_labels_hash)
check(indices_hash is not None, "hash labels: returns indices")
# Sample 0 (label same as 2) should NOT get 2 as negative
check(task_labels_hash[indices_hash[0]] != task_labels_hash[0],
      f"  sample 0 ('Pick up the red block'): negative is different task")

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"  Contrastive Loss Test: {passed} PASSED / {failed} FAILED")
print(f"{'='*60}")
if errors:
    print("  Failed:")
    for e in errors:
        print(f"    - {e}")

sys.exit(0 if failed == 0 else 1)
