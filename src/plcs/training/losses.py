"""Loss functions for PLCS training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def position_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute position loss (L1 or smooth L1).

    Args:
        pred: Predicted position, shape (B, 3).
        target: Target position, shape (B, 3).
        reduction: Reduction method ('mean', 'sum', 'none').

    Returns:
        Tensor: Position loss.

    """
    return nn.functional.smooth_l1_loss(pred, target, reduction=reduction)


def rotation_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute rotation loss for (sin, cos) representation.

    Uses cosine similarity loss which is appropriate for unit vectors.

    Args:
        pred: Predicted (sin, cos), shape (B, 2). Should be normalized.
        target: Target (sin, cos), shape (B, 2).
        reduction: Reduction method.

    Returns:
        Tensor: Rotation loss (1 - cosine similarity).

    """
    # Normalize predictions (should already be normalized, but ensure)
    pred_norm = nn.functional.normalize(pred, dim=-1)

    # Cosine similarity: dot product of unit vectors
    cos_sim = (pred_norm * target).sum(dim=-1)

    # Loss is 1 - cos_sim (0 when aligned, 2 when opposite)
    loss = 1.0 - cos_sim

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def angular_error(pred: Tensor, target: Tensor) -> Tensor:
    """Compute angular error in radians.

    Args:
        pred: Predicted (sin, cos), shape (B, 2).
        target: Target (sin, cos), shape (B, 2).

    Returns:
        Tensor: Angular error in radians, shape (B,).

    """
    # Convert to angles
    pred_angle = torch.atan2(pred[:, 0], pred[:, 1])
    target_angle = torch.atan2(target[:, 0], target[:, 1])

    # Angular difference (handle wraparound)
    diff = pred_angle - target_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))

    return diff.abs()


class PLCSLoss(nn.Module):
    """Combined loss for PLCS training.

    Combines position and rotation losses with configurable weights.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> None:
        """Initialize the loss module.

        Args:
            position_weight: Weight for position loss.
            rotation_weight: Weight for rotation loss.

        """
        super().__init__()
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight

    def forward(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            pred_position: Predicted position, shape (B, 3).
            pred_rotation: Predicted rotation, shape (B, 2).
            target_position: Target position, shape (B, 3).
            target_rotation: Target rotation, shape (B, 2).

        Returns:
            dict: Dictionary with 'total', 'position', and 'rotation' losses.

        """
        pos_loss = position_loss(pred_position, target_position)
        rot_loss = rotation_loss(pred_rotation, target_rotation)

        total = self.position_weight * pos_loss + self.rotation_weight * rot_loss

        return {
            "total": total,
            "position": pos_loss,
            "rotation": rot_loss,
        }
