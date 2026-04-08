#!/bin/bash

# cd /home/user/VLA_ws/new-vla/starVLA-AP
# conda activate libero

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/home/user/VLA_ws/new-vla/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/user/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

# headless rendering
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
# your_ckpt=../.playground/Pretrained_models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors
your_ckpt=./results/Checkpoints/0325_libero4in1_qwen35gr00t/final_model/pytorch_model.pt
# your_ckpt=../.playground/Pretrained_models/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
