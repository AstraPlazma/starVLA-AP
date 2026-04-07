"""
InfoNCE Contrastive Loss 单元测试
验证 Contr_ActionHeader 的 MSE + InfoNCE 损失功能。
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


def make_config(contrastive_weight=0.05, temperature=0.1):
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
                'infonce_temperature': temperature,
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
    f"weight=0: loss identical with/without labels ({out_no_label['loss'].item():.6f})"
)
check(out_no_label["contrastive_active"] == False, "weight=0: contrastive_active=False")

# =====================================================================
# 2. 不同 label 时，InfoNCE 生效
# =====================================================================
header("2. Different labels -> InfoNCE active")

head = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.05))

labels_diff = torch.tensor([0, 1, 2, 3])
torch.manual_seed(42)
out_contr = head(vl, actions, task_labels=labels_diff)

torch.manual_seed(42)
out_pure = head_mse(vl, actions)

check(torch.isfinite(out_contr["loss"]), f"total loss finite: {out_contr['loss'].item():.6f}")
check(out_contr["contrastive_active"] == True, "contrastive_active=True")
check(out_contr["infonce_loss"].item() > 0, f"infonce_loss > 0: {out_contr['infonce_loss'].item():.6f}")
check(out_contr["loss"].item() > out_pure["loss"].item(),
      f"MSE+InfoNCE > pure MSE ({out_contr['loss'].item():.6f} > {out_pure['loss'].item():.6f})")

# =====================================================================
# 3. loss 恒非负
# =====================================================================
header("3. Loss always non-negative")

head_strong = ContrFlowmatchingActionHead(make_config(contrastive_weight=1.0))

for seed in range(10):
    torch.manual_seed(seed)
    vl_t = torch.randn(B, L_vl, D_vl)
    act_t = torch.randn(B, T, D_action)
    lab_t = torch.tensor([seed % 4, (seed + 1) % 4, (seed + 2) % 4, (seed + 3) % 4])
    out_t = head_strong(vl_t, act_t, task_labels=lab_t)
    if out_t["contrastive_active"]:
        check(out_t["loss"].item() >= 0,
              f"seed={seed}: loss={out_t['loss'].item():.6f} >= 0")

# =====================================================================
# 4. 全同 label 退化为纯 MSE + 警告
# =====================================================================
header("4. All same labels -> fallback to MSE + warning")

labels_same = torch.tensor([5, 5, 5, 5])

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    torch.manual_seed(42)
    out_same = head(vl, actions, task_labels=labels_same)
    warning_fired = any("same task label" in str(warning.message) for warning in w)
    check(warning_fired, "Warning printed for all-same-label batch")

check(out_same["contrastive_active"] == False, "contrastive_active=False for all-same")
check(out_same["infonce_loss"].item() == 0.0, "infonce_loss=0 for all-same")

# =====================================================================
# 5. repeated_diffusion_steps 下 label 正确传播
# =====================================================================
header("5. Repeated diffusion steps")

repeated = 4
labels_base = torch.tensor([10, 20, 30, 40])
labels_repeated = labels_base.repeat(repeated)
vl_rep = vl.repeat(repeated, 1, 1)
actions_rep = actions.repeat(repeated, 1, 1)

out_rep = head(vl_rep, actions_rep, task_labels=labels_repeated)
check(torch.isfinite(out_rep["loss"]), f"repeated batch loss finite: {out_rep['loss'].item():.6f}")
check(out_rep["contrastive_active"] == True, "repeated: contrastive_active=True")
check(out_rep["loss"].item() >= 0, f"repeated: loss >= 0: {out_rep['loss'].item():.6f}")

# =====================================================================
# 6. 反向传播正常
# =====================================================================
header("6. Backward pass")

head_bp = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.05))
vl_bp = torch.randn(B, L_vl, D_vl)
act_bp = torch.randn(B, T, D_action)
labels_bp = torch.tensor([0, 1, 0, 2])

out_bp = head_bp(vl_bp, act_bp, task_labels=labels_bp)
out_bp["loss"].backward()

has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_bp.parameters())
check(has_grad, "Gradients flow through InfoNCE loss")

has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in head_bp.parameters())
check(not has_nan, "No NaN gradients")

# =====================================================================
# 7. task_labels=None -> no contrastive
# =====================================================================
header("7. task_labels=None -> no contrastive")

head_w = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.1))

torch.manual_seed(42)
out_none = head_w(vl, actions, task_labels=None)

check(torch.isfinite(out_none["loss"]), f"task_labels=None: loss finite ({out_none['loss'].item():.6f})")
check(out_none["contrastive_active"] == False, "task_labels=None: contrastive_active=False")
check(out_none["infonce_loss"].item() == 0.0, "task_labels=None: infonce_loss=0")

# =====================================================================
# 8. InfoNCE 利用全部负样本（不只一个）
# =====================================================================
header("8. InfoNCE uses all negatives")

# With 4 different labels, each sample has 3 negatives
labels_all_diff = torch.tensor([0, 1, 2, 3])
torch.manual_seed(42)
out_4neg = head(vl, actions, task_labels=labels_all_diff)

# With 2 labels (2 same, 2 same), each sample has 2 negatives
labels_2groups = torch.tensor([0, 0, 1, 1])
torch.manual_seed(42)
out_2neg = head(vl, actions, task_labels=labels_2groups)

# More negatives should generally give different InfoNCE value
check(
    abs(out_4neg["infonce_loss"].item() - out_2neg["infonce_loss"].item()) > 1e-4,
    f"InfoNCE differs with different negative counts ({out_4neg['infonce_loss'].item():.4f} vs {out_2neg['infonce_loss'].item():.4f})"
)

# =====================================================================
# 9. Temperature 影响 InfoNCE
# =====================================================================
header("9. Temperature effect")

head_cold = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.05, temperature=0.01))
head_warm = ContrFlowmatchingActionHead(make_config(contrastive_weight=0.05, temperature=1.0))

# Copy same weights
head_warm.load_state_dict(head_cold.state_dict())

torch.manual_seed(42)
out_cold = head_cold(vl, actions, task_labels=labels_diff)
torch.manual_seed(42)
out_warm = head_warm(vl, actions, task_labels=labels_diff)

check(
    out_cold["infonce_loss"].item() != out_warm["infonce_loss"].item(),
    f"Temperature affects InfoNCE (τ=0.01: {out_cold['infonce_loss'].item():.4f}, τ=1.0: {out_warm['infonce_loss'].item():.4f})"
)

# =====================================================================
# 10. Hash-based labels
# =====================================================================
header("10. Hash-based labels from instructions")

instructions = [
    "Pick up the red block",
    "Place the cup on the shelf",
    "Pick up the red block",
    "Open the drawer",
]

task_labels_hash = torch.tensor([hash(inst) % (2**31) for inst in instructions], dtype=torch.long)
check(task_labels_hash[0] == task_labels_hash[2], "same instruction -> same label")
check(task_labels_hash[0] != task_labels_hash[1], "different instruction -> different label")

out_hash = head(vl, actions, task_labels=task_labels_hash)
check(out_hash["contrastive_active"] == True, "hash labels: contrastive active")
check(out_hash["loss"].item() >= 0, f"hash labels: loss >= 0: {out_hash['loss'].item():.6f}")

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"  InfoNCE Loss Test: {passed} PASSED / {failed} FAILED")
print(f"{'='*60}")
if errors:
    print("  Failed:")
    for e in errors:
        print(f"    - {e}")

sys.exit(0 if failed == 0 else 1)
