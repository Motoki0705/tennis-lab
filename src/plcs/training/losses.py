"""Loss functions for PLCS training.

Supports both frame-level and sequence-level losses, including temporal
consistency for sequential models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class TemporalLossConfig:
    """Configuration for temporal consistency loss.

    Attributes:
        order: 1 => velocity smoothness, 2 => acceleration smoothness.
        robust: If True, use SmoothL1 (Huber) for stability against outliers.

    """

    order: int = 2
    robust: bool = True


@dataclass(frozen=True)
class PLCSLossConfig:
    """Configuration for PLCS loss weights.

    Attributes:
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss.
        temporal_weight: Weight for temporal consistency loss (0 = disabled).
        temporal: Temporal loss configuration.

    """

    position_weight: float = 1.0
    rotation_weight: float = 1.0
    temporal_weight: float = 0.0
    temporal: TemporalLossConfig | None = None

    @classmethod
    def from_dict(cls, cfg: dict) -> PLCSLossConfig:
        """Create config from dictionary (e.g., loaded from YAML).

        Args:
            cfg: Configuration dictionary.

        Returns:
            PLCSLossConfig instance.

        """
        temporal_dict = cfg.get("temporal", {})
        temporal_cfg = TemporalLossConfig(
            order=temporal_dict.get("order", 2),
            robust=temporal_dict.get("robust", True),
        )
        return cls(
            position_weight=cfg.get("position_weight", 1.0),
            rotation_weight=cfg.get("rotation_weight", 1.0),
            temporal_weight=cfg.get("temporal_weight", 0.0),
            temporal=temporal_cfg,
        )


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


def temporal_consistency_loss(
    pred_position: Tensor,
    target_position: Tensor,
    cfg: TemporalLossConfig,
) -> Tensor:
    """Compute temporal consistency loss for position sequences.

    Encourages smooth predictions by penalizing high-order derivatives.

    Args:
        pred_position: Predicted positions, shape (B, T, 3).
        target_position: Target positions, shape (B, T, 3).
        cfg: Temporal loss configuration.

    Returns:
        Tensor: Temporal consistency loss (scalar).

    """
    if pred_position.dim() == 2:
        # Single frame, no temporal loss
        return pred_position.new_zeros(())

    # Compute velocity (1st order difference)
    pred_vel = pred_position[:, 1:, :] - pred_position[:, :-1, :]  # (B, T-1, 3)
    target_vel = target_position[:, 1:, :] - target_position[:, :-1, :]

    if cfg.order == 1:
        # Velocity smoothness: pred velocity should match target velocity
        diff = pred_vel - target_vel
    else:
        # Acceleration smoothness (order=2): penalize 2nd order differences
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]  # (B, T-2, 3)
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        diff = pred_acc - target_acc

    if cfg.robust:
        return nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff))
    return diff.pow(2).mean()


class PLCSLoss(nn.Module):
    """Combined loss for PLCS training.

    Combines position, rotation, and optional temporal consistency losses.
    Supports both frame-level (B, 3) and sequence-level (B, T, 3) inputs.
    """

    def __init__(
        self,
        config: PLCSLossConfig | None = None,
        *,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> None:
        """Initialize the loss module.

        Args:
            config: Loss configuration (preferred). If provided, overrides
                position_weight and rotation_weight.
            position_weight: Weight for position loss (legacy parameter).
            rotation_weight: Weight for rotation loss (legacy parameter).

        """
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PLCSLossConfig(
                position_weight=position_weight,
                rotation_weight=rotation_weight,
                temporal_weight=0.0,
                temporal=TemporalLossConfig(),
            )

    def forward(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Supports both frame-level and sequence-level inputs:
            - Frame-level: (B, 3), (B, 2)
            - Sequence-level: (B, T, 3), (B, T, 2)

        Args:
            pred_position: Predicted position.
            pred_rotation: Predicted rotation.
            target_position: Target position.
            target_rotation: Target rotation.

        Returns:
            dict: Dictionary with 'total', 'position', 'rotation', and
                'temporal' (if enabled) losses.

        """
        zero = pred_position.new_zeros(())

        # Position loss
        pos_loss = position_loss(pred_position, target_position)

        # Rotation loss
        rot_loss = rotation_loss(pred_rotation, target_rotation)

        total = (
            self.config.position_weight * pos_loss
            + self.config.rotation_weight * rot_loss
        )

        # Temporal consistency loss (for sequences)
        if self.config.temporal_weight > 0.0 and self.config.temporal is not None:
            temp_loss = temporal_consistency_loss(
                pred_position, target_position, self.config.temporal
            )
            total = total + self.config.temporal_weight * temp_loss
        else:
            temp_loss = zero

        return {
            "total": total,
            "position": pos_loss,
            "rotation": rot_loss,
            "temporal": temp_loss,
        }
