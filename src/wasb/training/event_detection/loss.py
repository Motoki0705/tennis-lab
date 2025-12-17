"""Loss utilities for WASB trajectory event detection training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class EventDetectionLossConfig:
    ignore_index: int = -100
    label_smoothing: float = 0.0
    event_boost: float = 1.0
    background_weight_scale: float = 1.0


def event_detection_loss(
    *,
    logits: Tensor,
    target: Tensor,
    cfg: EventDetectionLossConfig,
    class_weights: Tensor | None = None,
) -> Tensor:
    b, t, c = logits.shape
    logits_flat = logits.reshape(b * t, c)
    target_flat = target.reshape(b * t)
    valid = target_flat != int(cfg.ignore_index)

    weights = class_weights
    if weights is not None and float(cfg.background_weight_scale) != 1.0:
        weights = weights.clone()
        weights[0] = weights[0] * float(cfg.background_weight_scale)

    loss_flat = F.cross_entropy(
        logits_flat,
        target_flat,
        weight=weights,
        ignore_index=int(cfg.ignore_index),
        reduction="none",
        label_smoothing=float(cfg.label_smoothing),
    )
    if float(cfg.event_boost) != 1.0:
        is_event = (target_flat == 1) | (target_flat == 2)
        boost = torch.ones_like(loss_flat)
        boost[is_event] = float(cfg.event_boost)
        loss_flat = loss_flat * boost

    if not valid.any():
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return (loss_flat[valid]).mean()

