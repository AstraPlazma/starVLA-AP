#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/home/user/miniconda3/envs/newvla/bin/python
# your_ckpt=../.playground/Pretrained_models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors
your_ckpt=./results/Checkpoints/0402_libero4in1_qwen35gr00t/checkpoints/steps_20000_pytorch_model.pt

gpu_id=0,1,2,3,4
port=5699
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
