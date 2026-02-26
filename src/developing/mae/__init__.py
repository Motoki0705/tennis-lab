"""Masked Autoencoder (MAE) for Vision Transformers.

This module provides training infrastructure for MAE pre-training
on tennis domain videos. The pre-trained ViT encoder can be used
for downstream tasks like ball tracking, player pose estimation, etc.

Components:
    - models/: MAE encoder-decoder architecture
    - data/: Cached-batch pipeline (planning/producer/dataset)
    - training/: Lightning module for MAE training
    - configs/: Hydra configurations
    - scripts/: Training and evaluation scripts

Reference:
    - MAE: https://arxiv.org/abs/2111.06377
"""
