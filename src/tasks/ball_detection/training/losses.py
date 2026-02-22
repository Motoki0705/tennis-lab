"""Loss utilities for sequence ball detection training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def weighted_xy_loss(
    pred_xy: Tensor,
    target_xy: Tensor,
    *,
    weight: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Weighted masked L1 loss on normalized coordinates."""
    base = F.l1_loss(pred_xy, target_xy, reduction="none").sum(dim=-1)

    if valid_mask is not None:
        v = valid_mask.to(base.dtype)
        base = base * v

    if weight is not None:
        w = weight.to(base.dtype)
        if valid_mask is not None:
            w = w * valid_mask.to(base.dtype)
        denom = w.sum().clamp_min(1e-6)
        return (base * w).sum() / denom

    if valid_mask is not None:
        denom = valid_mask.to(base.dtype).sum().clamp_min(1e-6)
        return base.sum() / denom
    return base.mean()


def visibility_bce_loss(
    logits: Tensor,
    target_vis: Tensor,
    *,
    weight: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Binary cross entropy on visibility logits with optional mask/weights."""
    base = F.binary_cross_entropy_with_logits(logits, target_vis, reduction="none")

    if valid_mask is not None:
        base = base * valid_mask.to(base.dtype)

    if weight is not None:
        w = weight.to(base.dtype)
        if valid_mask is not None:
            w = w * valid_mask.to(base.dtype)
        denom = w.sum().clamp_min(1e-6)
        return (base * w).sum() / denom

    if valid_mask is not None:
        denom = valid_mask.to(base.dtype).sum().clamp_min(1e-6)
        return base.sum() / denom
    return base.mean()


def event_aware_weight(base_weight: Tensor, event_mask: Tensor | None, event_boost: float) -> Tensor:
    """Scale per-frame weights near event frames."""
    if event_mask is None:
        return base_weight
    return base_weight * (1.0 + event_mask.to(base_weight.dtype) * float(event_boost))
