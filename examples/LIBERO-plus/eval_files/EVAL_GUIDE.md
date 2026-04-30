# LIBERO-plus 评估指南

## 环境信息

| 项目 | 路径/值 |
|------|---------|
| starVLA 工作目录 | `/home/user/VLA_ws/new-vla/starVLA/starVLA_vlm` |
| LIBERO-plus 代码库 | `/home/user/VLA_ws/new-vla/LIBERO-plus` |
| 模型推理环境 | `/home/user/miniconda3/envs/newvla/bin/python` |
| LIBERO-plus 仿真环境 | `/home/user/miniconda3/envs/liberop/bin/python` |
| 默认 checkpoint | `./results/Checkpoints/0415_libero4in1_mem3d_20k/final_model/pytorch_model.pt` |

---

## 评估架构

LIBERO-plus 提供两条评估路径：

```
路径 A（推荐，Client-Server 模式）
  run_policy_server.sh  ──►  eval_libero.sh
       ↓                          ↓
  启动 WebSocket 推理服务器    并行评估4个任务套件
  (newvla 环境, port 9883)    (liberop 环境)

路径 B（大规模并行，直接加载模型）
  parallel_eval/eval_libero_in_one.sh  ──►  parallel_eval/eval_libero_model.py
       ↓
  按任务范围切片，多进程并行，无需服务器
```

**本地评估推荐路径 A**，路径 B 适合云平台大规模批量评估。

---

## 路径 A：Client-Server 模式（推荐）

### 第一步：启动策略服务器

```bash
# 在 starVLA_vlm 目录下执行
cd /home/user/VLA_ws/new-vla/starVLA/starVLA_vlm
bash examples/LIBERO-plus/eval_files/run_policy_server.sh
```

服务器启动后监听 `port 9883`，等待日志输出 `Server started` 后再进行下一步。

关键参数（[run_policy_server.sh](run_policy_server.sh)）：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `your_ckpt` | `./results/Checkpoints/0415_libero4in1_mem3d_20k/final_model/pytorch_model.pt` | 模型 checkpoint |
| `base_port` | `9883` | 服务监听端口 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6` | 使用的 GPU |
| `--use_bf16` | 已启用 | bfloat16 精度推理 |

### 第二步：运行评估

新开终端，在同一工作目录下执行：

```bash
cd /home/user/VLA_ws/new-vla/starVLA/starVLA_vlm
bash examples/LIBERO-plus/eval_files/eval_libero.sh
```

脚本并行评估4个任务套件，全部完成后自动退出。

关键参数（[eval_libero.sh](eval_libero.sh)）：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `LIBERO_HOME` | `/home/user/VLA_ws/new-vla/LIBERO-plus` | LIBERO-plus 代码库路径 |
| `LIBERO_Python` | `/home/user/miniconda3/envs/liberop/bin/python` | 仿真环境 Python |
| `your_ckpt` | `./results/Checkpoints/...` | 与服务器保持一致 |
| `output_dir` | `./results/libero_plus_eval` | 结果输出目录 |
| `base_port` | `9883` | 连接服务器端口 |
| `num_trials_per_task` | `1` | 每任务 rollout 次数（任务量大，保持1） |
| `MUJOCO_GL` | `osmesa` | 无头渲染 |

### 第三步：聚合结果

```bash
export LOG_DIR="./results/libero_plus_eval/logs/<timestamp>"
python examples/LIBERO-plus/eval_files/aggregate_results.py
```

结果输出至 `${LOG_DIR}/overall_results.json`。

---

## 路径 B：并行直接推理（大规模）

适用于需要对大量数据切片并行评估的场景（如云平台）。

```bash
# 单节点内并行评估某个任务套件的部分任务
bash examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_in_one.sh \
    <task_suite_name> <start_idx> <end_idx>

# 示例：评估 libero_goal 的前100个任务
bash examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_in_one.sh \
    libero_goal 0 100
```

关键参数（[parallel_eval/eval_libero_in_one.sh](parallel_eval/eval_libero_in_one.sh)）：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `tasks_per_gpu` | `3` | 节点内并行进程数 |
| `your_ckpt` | `path_to_checkpoint` | **需手动修改** |
| `output_dir` | `path_to_output_dir` | **需手动修改** |
| `unnorm_key` | `franka` | 动作反归一化键 |

聚合结果：

```bash
python examples/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py \
    --root_path ./results/libero_plus_eval
```

---

## 任务套件说明

| 套件名 | 任务数（LIBERO-plus） | max_steps | 说明 |
|--------|----------------------|-----------|------|
| `libero_goal` | ~2591 | 300 | 目标导向任务 |
| `libero_spatial` | ~2402 | 220 | 空间关系任务 |
| `libero_object` | ~2518 | 280 | 物体操作任务 |
| `libero_10` | ~2519 | 520 | 长序列任务 |

---

## 与 LIBERO（原版）的主要差异

| 项目 | LIBERO | LIBERO-plus |
|------|--------|-------------|
| 代码库 | `/home/user/VLA_ws/new-vla/LIBERO` | `/home/user/VLA_ws/new-vla/LIBERO-plus` |
| Python 环境 | `envs/libero` | `envs/liberop` |
| 服务器端口 | `5697` | `9883` |
| `num_trials_per_task` | `50` | `1`（任务量更大） |
| 任务数量 | 少 | 每套件约2400-2600 |
| 结果输出 | `logs/` | `results/libero_plus_eval/logs/` |

---

## 常见问题

**Q: 服务器启动后评估脚本连接失败？**
确认 `base_port` 两个脚本一致（均为 `9883`），且服务器已完全启动。

**Q: MuJoCo 渲染报错？**
确认已设置 `MUJOCO_GL=osmesa` 和 `PYOPENGL_PLATFORM=osmesa`，并安装了 `libOSMesa`。

**Q: 需要更换 checkpoint？**
同时修改 `run_policy_server.sh` 和 `eval_libero.sh` 中的 `your_ckpt`，保持一致。
