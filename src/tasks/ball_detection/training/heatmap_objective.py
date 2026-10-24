"""Gaussian targets and spatial-mean focal BCE shared by DPT and hybrid."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.tasks.base.training.losses import FocalBCEWithLogitsLoss
from src.utils.data.heatmaps import generate_gaussian_heatmaps


def heatmap_objective(logits: Tensor, batch: dict[str, Any], *, sigma_ratio: float = .012,
                      gamma: float = 2.) -> tuple[Tensor, Tensor]:
    """Return masked scalar loss and per-frame loss; all target/loss arithmetic is FP32."""
    if logits.ndim != 4 or not logits.is_floating_point() or min(logits.shape) < 1:
        raise ValueError("Heatmap logits must be a nonempty floating B,T,H,W tensor")
    device = logits.device
    positive = batch["position_valid"].to(device, non_blocking=True)
    supervised = batch["heatmap_valid"].to(device, non_blocking=True)
    uv = batch["uv"].to(device, non_blocking=True)
    if positive.dtype != torch.bool or supervised.dtype != torch.bool or positive.shape != logits.shape[:2] or supervised.shape != positive.shape:
        raise ValueError("Heatmap targets require boolean B,T masks aligned with logits")
    if bool((positive & ~supervised).any()):
        raise ValueError("Position supervision must imply heatmap supervision")
    if uv.shape != (*positive.shape, 2) or not uv.is_floating_point():
        raise ValueError("Heatmap coordinates require floating B,T,2 values")
    if not bool(torch.isfinite(uv[positive]).all()) or bool(((uv[positive] < 0) | (uv[positive] > 1)).any()):
        raise ValueError("Supervised positions must be finite normalized coordinates")
    uv = torch.where(positive[..., None], uv, 0.)
    target = generate_gaussian_heatmaps(tuple(logits.shape[-2:]), uv, sigma_ratio, visibility=positive)
    per_frame = FocalBCEWithLogitsLoss(gamma).elementwise(logits.float(), target).mean(dim=(-2, -1))
    count = supervised.sum()
    # A batch may contain only confirmed absence, but never only unknown frames.
    if not bool(count):
        raise ValueError("No supervised frames in heatmap batch")
    loss = torch.where(supervised, per_frame, 0.).sum() / count
    return loss, per_frame
