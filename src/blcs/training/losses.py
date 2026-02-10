"""Loss functions for BLCS training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.geometry.constants import COURT_COORD_SCALE_XYZ


def _masked_mean(loss: Tensor, mask: Tensor | None = None) -> Tensor:
    """Compute mean loss with an optional mask."""
    if mask is None:
        return loss.mean()

    mask_expanded = mask.to(dtype=loss.dtype)
    while mask_expanded.ndim < loss.ndim:
        mask_expanded = mask_expanded.unsqueeze(-1)
    mask_expanded = mask_expanded.expand_as(loss)
    return (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)


def trajectory_position_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute position loss for trajectory prediction.

    Uses Smooth L1 loss which is less sensitive to outliers.

    Args:
        pred: Predicted positions, shape (B, T, 3).
        target: Target positions, shape (B, T, 3).
        mask: Visibility mask, shape (B, T). 1 = valid, 0 = ignore.

    Returns:
        Tensor: Position loss.

    """
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return _masked_mean(loss, mask)


def velocity_loss(
    pred_pos: Tensor,
    target_pos: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute velocity consistency loss.

    Encourages predicted trajectory to have consistent velocities
    with the ground truth.

    Args:
        pred_pos: Predicted positions, shape (B, T, 3).
        target_pos: Target positions, shape (B, T, 3).
        mask: Visibility mask, shape (B, T).

    Returns:
        Tensor: Velocity loss.

    """
    # Compute finite differences (velocities)
    pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
    target_vel = target_pos[:, 1:] - target_pos[:, :-1]

    loss = nn.functional.smooth_l1_loss(pred_vel, target_vel, reduction="none")
    # Velocity is valid only when both adjacent frames are valid.
    vel_mask = mask[:, 1:] * mask[:, :-1] if mask is not None else None
    return _masked_mean(loss, vel_mask)


def smoothness_loss(
    pred_pos: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute trajectory smoothness loss.

    Penalizes high acceleration (second derivative) to encourage
    smooth trajectories.

    Args:
        pred_pos: Predicted positions, shape (B, T, 3).
        mask: Visibility mask, shape (B, T).

    Returns:
        Tensor: Smoothness loss.

    """
    # Compute second derivative (acceleration)
    vel = pred_pos[:, 1:] - pred_pos[:, :-1]
    accel = vel[:, 1:] - vel[:, :-1]

    loss = (accel**2).sum(dim=-1)  # (B, T-2)
    # Acceleration is valid only when three consecutive frames are valid.
    accel_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2] if mask is not None else None
    return _masked_mean(loss, accel_mask)


class BLCSLoss(nn.Module):
    """Combined loss for BLCS training.

    Combines position, velocity, and smoothness losses with configurable weights.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        velocity_weight: float = 0.1,
        smoothness_weight: float = 0.05,
    ) -> None:
        """Initialize the loss module.

        Args:
            position_weight: Weight for position loss.
            velocity_weight: Weight for velocity consistency loss.
            smoothness_weight: Weight for smoothness loss.

        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred_position: Tensor,
        target_position: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            pred_position: Predicted positions, shape (B, T, 3).
            target_position: Target positions, shape (B, T, 3).
            mask: Visibility mask, shape (B, T).

        Returns:
            dict: Dictionary with 'total', 'position', 'velocity', 'smoothness'.

        """
        # Position loss
        pos_loss = trajectory_position_loss(pred_position, target_position, mask)

        # Velocity consistency loss
        vel_loss = velocity_loss(pred_position, target_position, mask)

        # Smoothness loss
        smooth_loss = smoothness_loss(pred_position, mask)

        # Combined loss
        total = (
            self.position_weight * pos_loss
            + self.velocity_weight * vel_loss
            + self.smoothness_weight * smooth_loss
        )

        return {
            "total": total,
            "position": pos_loss,
            "velocity": vel_loss,
            "smoothness": smooth_loss,
        }
