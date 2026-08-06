"""Axis-balanced position loss primitives for BLCS tracking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def position_axis_weight_tensor(value: Sequence[float] | Tensor) -> Tensor:
    """Validate and convert three positive BLCS position-axis weights."""
    weights = torch.as_tensor(value, dtype=torch.float32)
    if weights.shape != (3,):
        raise ValueError("position_axis_weights must contain exactly 3 values.")
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("position_axis_weights must be finite and positive.")
    return weights


def weighted_position_axis_mean(values: Tensor, axis_weights: Tensor) -> Tensor:
    """Reduce the boundary-validated final XYZ axis using fixed weights."""
    weights = axis_weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum(-1) / weights.sum()


__all__ = ["position_axis_weight_tensor", "weighted_position_axis_mean"]
