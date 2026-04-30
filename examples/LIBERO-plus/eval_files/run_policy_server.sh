#!/bin/bash

your_ckpt=./results/Checkpoints/0427_libero4in1_mem3d_20k_8/final_model/pytorch_model.pt
base_port=9883
export star_vla_python=/home/user/miniconda3/envs/newvla/bin/python

export PYTHONPATH=$(pwd):${PYTHONPATH}

# export DEBUG=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16