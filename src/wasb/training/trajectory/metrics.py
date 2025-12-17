"""Metrics utilities for WASB trajectory completion training."""

from __future__ import annotations

import torch
from torch import Tensor

from src.wasb.training.trajectory.loss import masked_mean


def rmse_px_from_norm(
    *,
    pred_xy_norm: Tensor,
    target_xy_norm: Tensor,
    mask: Tensor,
    scale_xy_px: tuple[float, float] = (1920.0, 1080.0),
) -> Tensor:
    device = pred_xy_norm.device
    scale = torch.tensor(list(scale_xy_px), dtype=torch.float32, device=device)
    pred_px = pred_xy_norm * scale
    target_px = target_xy_norm * scale
    diff_px = pred_px - target_px
    sq = (diff_px * diff_px).sum(dim=-1)
    return torch.sqrt(masked_mean(sq, mask.to(dtype=torch.float32, device=device)))

