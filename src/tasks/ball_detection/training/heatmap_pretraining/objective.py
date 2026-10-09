from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.tasks.ball_detection.training.coordinate_images import decode_coordinate_jpegs
from src.tasks.base.training.losses import FocalBCEWithLogitsLoss
from src.utils.data.heatmaps import generate_gaussian_heatmaps


def image_input(batch: dict[str, Any], device: torch.device) -> Tensor:
    rgb = batch["rgb"].to(device, non_blocking=True) if "rgb" in batch else decode_coordinate_jpegs(batch, device)
    if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[1:3] != (32, 3):
        raise ValueError("Pretraining requires RGB uint8 B,32,3,H,W")
    return rgb


def heatmap_objective(logits: Tensor, batch: dict[str, Any], *, sigma_ratio: float = .012,
                      gamma: float = 2.) -> tuple[Tensor, Tensor]:
    """Return masked scalar loss and per-frame loss; all target/loss arithmetic is FP32."""
    device = logits.device
    positive = batch["position_valid"].to(device, non_blocking=True)
    supervised = batch["heatmap_valid"].to(device, non_blocking=True)
    uv = batch["uv"].to(device, non_blocking=True)
    uv = torch.where(positive[..., None], uv, 0.)
    target = generate_gaussian_heatmaps(tuple(logits.shape[-2:]), uv, sigma_ratio, visibility=positive)
    per_frame = FocalBCEWithLogitsLoss(gamma).elementwise(logits.float(), target).mean(dim=(-2, -1))
    count = supervised.sum()
    # Accepted windows contain at least eight observed frames; zero is a data error.
    if not bool(count):
        raise ValueError("No supervised frames in heatmap batch")
    loss = torch.where(supervised, per_frame, 0.).sum() / count
    return loss, per_frame
