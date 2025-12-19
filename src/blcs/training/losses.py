"""Loss functions for BLCS training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.geometry.constants import COURT_COORD_SCALE_XYZ

def trajectory_position_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute position loss for trajectory prediction.

    Uses Smooth L1 loss which is less sensitive to outliers.

    Args:
        pred: Predicted positions, shape (B, T, 3).
        target: Target positions, shape (B, T, 3).
        mask: Visibility mask, shape (B, T). 1 = valid, 0 = ignore.
        reduction: Reduction method ('mean', 'sum', 'none').

    Returns:
        Tensor: Position loss.

    """
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")

    if mask is not None:
        # Expand mask for 3D coords
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask_expanded

        if reduction == "mean":
            return loss.sum() / (mask_expanded.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
        return loss
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss


def velocity_loss(
    pred_pos: Tensor,
    target_pos: Tensor,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute velocity consistency loss.

    Encourages predicted trajectory to have consistent velocities
    with the ground truth.

    Args:
        pred_pos: Predicted positions, shape (B, T, 3).
        target_pos: Target positions, shape (B, T, 3).
        mask: Visibility mask, shape (B, T).
        reduction: Reduction method.

    Returns:
        Tensor: Velocity loss.

    """
    # Compute finite differences (velocities)
    pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
    target_vel = target_pos[:, 1:] - target_pos[:, :-1]

    loss = nn.functional.smooth_l1_loss(pred_vel, target_vel, reduction="none")

    if mask is not None:
        # Mask for velocity needs both frames to be valid
        vel_mask = mask[:, 1:] * mask[:, :-1]
        mask_expanded = vel_mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask_expanded

        if reduction == "mean":
            return loss.sum() / (mask_expanded.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
        return loss
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss


def smoothness_loss(
    pred_pos: Tensor,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute trajectory smoothness loss.

    Penalizes high acceleration (second derivative) to encourage
    smooth trajectories.

    Args:
        pred_pos: Predicted positions, shape (B, T, 3).
        mask: Visibility mask, shape (B, T).
        reduction: Reduction method.

    Returns:
        Tensor: Smoothness loss.

    """
    # Compute second derivative (acceleration)
    vel = pred_pos[:, 1:] - pred_pos[:, :-1]
    accel = vel[:, 1:] - vel[:, :-1]

    loss = (accel**2).sum(dim=-1)  # (B, T-2)

    if mask is not None:
        # Mask needs three consecutive frames
        accel_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        loss = loss * accel_mask

        if reduction == "mean":
            return loss.sum() / (accel_mask.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
        return loss
    else:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss


def position_error_meters(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    scale_xyz: tuple[float, float, float] = COURT_COORD_SCALE_XYZ,
) -> Tensor:
    """Compute position error in meters.

    Args:
        pred: Predicted positions (normalized), shape (B, T, 3).
        target: Target positions (normalized), shape (B, T, 3).
        mask: Visibility mask, shape (B, T).

    Returns:
        Tensor: Mean position error in meters.

    """
    # Denormalize to meters
    scale = torch.tensor(
        list(scale_xyz),
        device=pred.device,
    )
    pred_m = pred * scale
    target_m = target * scale

    # Compute Euclidean distance
    error = torch.sqrt(((pred_m - target_m) ** 2).sum(dim=-1) + 1e-8)  # (B, T)

    if mask is not None:
        return (error * mask).sum() / (mask.sum() + 1e-8)
    return error.mean()


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
