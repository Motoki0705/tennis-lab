"""Angle and vector geometry helpers (torch).

Consolidates the wrapped-angle / safe-normalize / signed-angle primitives that
were defined privately inside ``plcs`` losses and metrics (and re-derived in
analysis scripts).
"""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_vector(v: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize vectors along the last dimension with a safe denominator."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def wrapped_angle_diff(pred_angle: Tensor, target_angle: Tensor) -> Tensor:
    """Return the signed wrapped angle difference in ``[-pi, pi]``."""
    diff = pred_angle - target_angle
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def angular_error(pred: Tensor, target: Tensor) -> Tensor:
    """Compute wrapped angular error in radians between two ``(cos, sin)`` pairs.

    Args:
        pred: Predicted ``(cos, sin)``, shape ``(..., 2)``.
        target: Target ``(cos, sin)``, shape ``(..., 2)``.

    Returns:
        Tensor: Absolute angular error in radians, shape ``(...)``.
    """
    pred_angle = torch.atan2(pred[..., 1], pred[..., 0])
    target_angle = torch.atan2(target[..., 1], target[..., 0])
    diff = pred_angle - target_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff.abs()


def signed_angle_around_axis(
    v1: Tensor,
    v2: Tensor,
    axis: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the signed angle from ``v1`` to ``v2`` around ``axis``.

    The vectors are first projected onto the plane perpendicular to ``axis``.
    Useful for measuring body twist around the torso axis.

    Args:
        v1: First vector, shape ``(..., 3)``.
        v2: Second vector, shape ``(..., 3)``.
        axis: Rotation axis, shape ``(..., 3)``.
        eps: Numerical stability epsilon.

    Returns:
        Tensor: Signed angle in radians, shape ``(...)``.
    """
    axis = normalize_vector(axis, eps=eps)

    v1_proj = v1 - (v1 * axis).sum(dim=-1, keepdim=True) * axis
    v2_proj = v2 - (v2 * axis).sum(dim=-1, keepdim=True) * axis

    v1_proj = normalize_vector(v1_proj, eps=eps)
    v2_proj = normalize_vector(v2_proj, eps=eps)

    x = (v1_proj * v2_proj).sum(dim=-1)
    y = (torch.cross(v1_proj, v2_proj, dim=-1) * axis).sum(dim=-1)
    return torch.atan2(y, x)


__all__ = [
    "angular_error",
    "normalize_vector",
    "signed_angle_around_axis",
    "wrapped_angle_diff",
]
