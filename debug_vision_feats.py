#!/usr/bin/env python3
"""Debug script to check vision_feats dimensions"""
import sys
sys.path.insert(0, '/home/user/VLA_ws/new-vla/starVLA-AP')

from omegaconf import OmegaConf
from starVLA.model.framework.QwenMem import QwenMem
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
from torch.utils.data import DataLoader
import torch

# Load config
cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")

# Create dataset
dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
dataloader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=collate_fn)

# Get one batch
batch = next(iter(dataloader))

print(f"Batch size: {len(batch)}")
print(f"Number of images per sample: {[len(ex['image']) for ex in batch]}")
print(f"Total images: {sum(len(ex['image']) for ex in batch)}")
