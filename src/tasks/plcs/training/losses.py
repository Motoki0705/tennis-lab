"""Loss functions for PLCS training.

Supports frame-level and sequence-level losses.
Temporal consistency is enforced by the GAN discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.utils.pose_geometry import world_pose_to_canonical_pose
from src.utils.tensor_utils import masked_mean, normalize_padding_mask


@dataclass(frozen=True)
class PLCSLossConfig:
    """Configuration for PLCS loss weights.

    Attributes:
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss.
        canonical_pose_weight: Weight for canonical pose loss.

    """

    position_weight: float = 1.0
    rotation_weight: float = 1.0
    canonical_pose_weight: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict) -> PLCSLossConfig:
        """Create config from dictionary (e.g., loaded from YAML).

        Args:
            cfg: Configuration dictionary.

        Returns:
            PLCSLossConfig instance.

        """
        return cls(
            position_weight=cfg.get("position_weight", 1.0),
            rotation_weight=cfg.get("rotation_weight", 1.0),
            canonical_pose_weight=float(cfg.get("canonical_pose_weight", 0.0)),
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
    """Compute rotation loss for (cos, sin) representation.

    Uses cosine similarity loss which is appropriate for unit vectors.

    Args:
        pred: Predicted (cos, sin), shape (B, 2). Should be normalized.
        target: Target (cos, sin), shape (B, 2).
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
        pred: Predicted (cos, sin), shape (B, 2).
        target: Target (cos, sin), shape (B, 2).

    Returns:
        Tensor: Angular error in radians, shape (B,).

    """
    # Convert to angles
    pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])

    # Angular difference (handle wraparound)
    diff = pred_angle - target_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))

    return diff.abs()


class PLCSLoss(nn.Module):
    """Combined loss for PLCS training.

    Combines position and rotation losses.
    Supports both frame-level (B, 3) and sequence-level (B, T, 3) inputs.
    Temporal consistency is enforced by the GAN discriminator.
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
            )

    def forward(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        *,
        pred_canonical_pose: Tensor | None = None,
        target_human_kp_3d: Tensor | None = None,
        human_mask: Tensor | None = None,
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
            dict: Dictionary with 'total', 'position', 'rotation', 'canonical_pose'.

        """
        zero = pred_position.new_zeros(())
        frame_mask = normalize_padding_mask(human_mask, flatten=False)

        # Position loss (optionally mask padded frames)
        if pred_position.dim() == 3 and frame_mask is not None:
            per_elem = nn.functional.smooth_l1_loss(
                pred_position, target_position, reduction="none"
            ).mean(dim=-1)  # (B, T)
            pos_loss = masked_mean(per_elem, frame_mask, binarize=True, denom_min=1.0)
        else:
            pos_loss = position_loss(pred_position, target_position)

        # Rotation loss (optionally mask padded frames)
        if pred_rotation.dim() == 3 and frame_mask is not None:
            pred_norm = nn.functional.normalize(pred_rotation, dim=-1)
            cos_sim = (pred_norm * target_rotation).sum(dim=-1)  # (B, T)
            per_frame = 1.0 - cos_sim
            rot_loss = masked_mean(per_frame, frame_mask, binarize=True, denom_min=1.0)
        else:
            rot_loss = rotation_loss(pred_rotation, target_rotation)

        total = (
            self.config.position_weight * pos_loss
            + self.config.rotation_weight * rot_loss
        )
        canonical_pose_loss = zero

        if pred_canonical_pose is not None and target_human_kp_3d is not None:
            target_canonical_pose = world_pose_to_canonical_pose(
                target_human_kp_3d,
                target_position,
                target_rotation,
            )
            per_frame = nn.functional.smooth_l1_loss(
                pred_canonical_pose,
                target_canonical_pose,
                reduction="none",
            ).mean(dim=(-1, -2))
            if frame_mask is not None and per_frame.shape == frame_mask.shape:
                canonical_pose_loss = masked_mean(per_frame, frame_mask, binarize=True, denom_min=1.0)
            else:
                canonical_pose_loss = per_frame.mean()
            total = total + self.config.canonical_pose_weight * canonical_pose_loss
        elif pred_canonical_pose is not None and self.config.canonical_pose_weight > 0.0:
            raise ValueError(
                "target_human_kp_3d is required when canonical_pose_weight > 0 and "
                "pred_canonical_pose is provided."
            )

        return {
            "total": total,
            "position": pos_loss,
            "rotation": rot_loss,
            "canonical_pose": canonical_pose_loss,
        }
