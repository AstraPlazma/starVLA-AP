#!/usr/bin/env bash
# =============================================================================
# starVLA 训练启动脚本（train_starvla.py）
# - 作用：使用 accelerate + DeepSpeed 启动训练。
# - 注意：仅在“Paths to edit”区域修改变量值（模型、数据路径、run id 等）。
# =============================================================================

# -------------------------
# 通信 / NCCL 设置（按硬件/网络环境调整）
# -------------------------
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# 保证在通信异常或同步操作时的可见性与超时控制
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # 单位：秒（此处设为 1 小时）


# -------------------------
# 禁用 W&B 自动在线记录（按需启用）
# -------------------------
export WANDB_MODE=disabled


# -------------------------
# Paths to edit / 运行配置（根据实际环境修改）
# -------------------------
Framework_name=QwenOFT
base_vlm=StarVLA/Qwen3-VL-4B-Instruct-Action
action_input_dim=2560
oxe_data_root=playground/Datasets/OXE_LEROBOT
data_mix=bridge_rt_1
run_root_dir=./playground/Checkpoints
run_id=1004_starvla_qwenoft_oxe


# -------------------------
# 输出目录准备（创建目录并备份本脚本）
# -------------------------
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/


# -------------------------
# 启动：accelerate + DeepSpeed（单机多卡示例）
# - 若在多节点环境运行，请使用下方 multi-node 示例并填入集群变量。
# -------------------------
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_hidden_dim ${action_input_dim} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jinhuiye \
  # --is_debug True


# -------------------------
# multi-node 启动示例（注释示例，按需取消注释并设置集群变量）
# - 在 Slurm 或自建多机环境中运行时使用。
# -------------------------
# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
#   --framework.framework_py QwenGR00T \
#   --framework.qwenvl.base_vlm microsoft/Florence-2-large \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project your_project \
#   --wandb_entity your_name
# =============================================================================