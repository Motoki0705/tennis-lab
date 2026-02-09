"""Loss functions for PLCS training.

Supports frame-level and sequence-level losses, including temporal consistency
terms for sequential models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class TemporalTermConfig:
    """Configuration for a single temporal consistency term.

    Attributes:
        weight: Weight applied to this term in the total loss.
        order: 1 => velocity (1st difference), 2 => acceleration (2nd difference).
        robust: If True, use SmoothL1 (Huber) loss; else use MSE.

    """

    weight: float = 0.0
    order: int = 2
    robust: bool = True


@dataclass(frozen=True)
class TemporalTermsConfig:
    """Configuration for temporal consistency terms.

    Terms:
        - position_gt: match GT velocity/acceleration
        - position_inertia: encourage constant velocity/acceleration-free motion
        - rotation_gt: match GT angular velocity/acceleration
        - rotation_inertia: encourage constant angular velocity
    """

    position_gt: TemporalTermConfig = TemporalTermConfig()
    position_inertia: TemporalTermConfig = TemporalTermConfig()
    rotation_gt: TemporalTermConfig = TemporalTermConfig()
    rotation_inertia: TemporalTermConfig = TemporalTermConfig()


@dataclass(frozen=True)
class PLCSLossConfig:
    """Configuration for PLCS loss weights.

    Attributes:
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss.
        temporal: Temporal term configuration.

    """

    position_weight: float = 1.0
    rotation_weight: float = 1.0
    temporal: TemporalTermsConfig = TemporalTermsConfig()

    @classmethod
    def from_dict(cls, cfg: dict) -> PLCSLossConfig:
        """Create config from dictionary (e.g., loaded from YAML).

        Args:
            cfg: Configuration dictionary.

        Returns:
            PLCSLossConfig instance.

        """
        if "temporal_weight" in cfg:
            raise ValueError(
                "Legacy `temporal_weight` is no longer supported. "
                "Use `temporal.position_gt.weight` etc. instead."
            )

        temporal_dict = cfg.get("temporal")
        temporal_dict = temporal_dict if isinstance(temporal_dict, dict) else {}

        if any(k in temporal_dict for k in ["order", "robust"]):
            raise ValueError(
                "Legacy `temporal: {order, robust}` is no longer supported. "
                "Use `temporal.position_gt` / `temporal.rotation_gt` etc. instead."
            )

        def _parse_term(d: dict | None) -> TemporalTermConfig:
            d = d or {}
            return TemporalTermConfig(
                weight=float(d.get("weight", 0.0)),
                order=int(d.get("order", 2)),
                robust=bool(d.get("robust", True)),
            )

        temporal_terms_cfg = TemporalTermsConfig(
            position_gt=_parse_term(temporal_dict.get("position_gt")),
            position_inertia=_parse_term(temporal_dict.get("position_inertia")),
            rotation_gt=_parse_term(temporal_dict.get("rotation_gt")),
            rotation_inertia=_parse_term(temporal_dict.get("rotation_inertia")),
        )

        # Validate supported orders early
        for name, term in [
            ("position_gt", temporal_terms_cfg.position_gt),
            ("position_inertia", temporal_terms_cfg.position_inertia),
            ("rotation_gt", temporal_terms_cfg.rotation_gt),
            ("rotation_inertia", temporal_terms_cfg.rotation_inertia),
        ]:
            if term.order not in (1, 2):
                raise ValueError(
                    f"temporal.{name}.order must be 1 or 2 (got {term.order})"
                )

        return cls(
            position_weight=cfg.get("position_weight", 1.0),
            rotation_weight=cfg.get("rotation_weight", 1.0),
            temporal=temporal_terms_cfg,
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


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def _wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _sequence_frame_mask(seq_mask: Tensor | None, *, order: int) -> Tensor | None:
    if seq_mask is None:
        return None
    if order == 1:
        return seq_mask[:, 1:] & seq_mask[:, :-1]
    return seq_mask[:, 2:] & seq_mask[:, 1:-1] & seq_mask[:, :-2]


def position_temporal_match_loss(
    pred_position: Tensor,
    target_position: Tensor,
    cfg: TemporalTermConfig,
    *,
    seq_mask: Tensor | None = None,
) -> Tensor:
    if pred_position.dim() == 2:
        return pred_position.new_zeros(())

    pred_vel = pred_position[:, 1:, :] - pred_position[:, :-1, :]  # (B, T-1, 3)
    target_vel = target_position[:, 1:, :] - target_position[:, :-1, :]

    if cfg.order == 1:
        diff = pred_vel - target_vel  # (B, T-1, 3)
        mask = _sequence_frame_mask(seq_mask, order=1)
    else:
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]  # (B, T-2, 3)
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        diff = pred_acc - target_acc
        mask = _sequence_frame_mask(seq_mask, order=2)

    if mask is not None and mask.sum().item() == 0:
        return pred_position.new_zeros(())

    if cfg.robust:
        per_elem = nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    else:
        per_elem = diff.pow(2)

    per_step = per_elem.mean(dim=-1)  # (B, T-1) or (B, T-2)
    return _masked_mean(per_step, mask)


def position_temporal_inertia_loss(
    pred_position: Tensor,
    cfg: TemporalTermConfig,
    *,
    seq_mask: Tensor | None = None,
) -> Tensor:
    if pred_position.dim() == 2:
        return pred_position.new_zeros(())

    pred_vel = pred_position[:, 1:, :] - pred_position[:, :-1, :]  # (B, T-1, 3)

    if cfg.order == 1:
        diff = pred_vel
        mask = _sequence_frame_mask(seq_mask, order=1)
    else:
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]  # (B, T-2, 3)
        diff = pred_acc
        mask = _sequence_frame_mask(seq_mask, order=2)

    if mask is not None and mask.sum().item() == 0:
        return pred_position.new_zeros(())

    if cfg.robust:
        per_elem = nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    else:
        per_elem = diff.pow(2)

    per_step = per_elem.mean(dim=-1)
    return _masked_mean(per_step, mask)


def rotation_temporal_match_loss(
    pred_rotation: Tensor,
    target_rotation: Tensor,
    cfg: TemporalTermConfig,
    *,
    seq_mask: Tensor | None = None,
) -> Tensor:
    if pred_rotation.dim() == 2:
        return pred_rotation.new_zeros(())

    yaw_pred = torch.atan2(pred_rotation[..., 1], pred_rotation[..., 0])  # (B, T)
    yaw_target = torch.atan2(target_rotation[..., 1], target_rotation[..., 0])

    dyaw_pred = _wrap_angle(yaw_pred[:, 1:] - yaw_pred[:, :-1])  # (B, T-1)
    dyaw_target = _wrap_angle(yaw_target[:, 1:] - yaw_target[:, :-1])

    if cfg.order == 1:
        diff = _wrap_angle(dyaw_pred - dyaw_target)  # (B, T-1)
        mask = _sequence_frame_mask(seq_mask, order=1)
    else:
        ddyaw_pred = _wrap_angle(dyaw_pred[:, 1:] - dyaw_pred[:, :-1])  # (B, T-2)
        ddyaw_target = _wrap_angle(dyaw_target[:, 1:] - dyaw_target[:, :-1])
        diff = _wrap_angle(ddyaw_pred - ddyaw_target)
        mask = _sequence_frame_mask(seq_mask, order=2)

    if mask is not None and mask.sum().item() == 0:
        return pred_rotation.new_zeros(())

    if cfg.robust:
        per_step = nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    else:
        per_step = diff.pow(2)

    return _masked_mean(per_step, mask)


def rotation_temporal_inertia_loss(
    pred_rotation: Tensor,
    cfg: TemporalTermConfig,
    *,
    seq_mask: Tensor | None = None,
) -> Tensor:
    if pred_rotation.dim() == 2:
        return pred_rotation.new_zeros(())

    yaw_pred = torch.atan2(pred_rotation[..., 1], pred_rotation[..., 0])  # (B, T)
    dyaw_pred = _wrap_angle(yaw_pred[:, 1:] - yaw_pred[:, :-1])  # (B, T-1)

    if cfg.order == 1:
        diff = dyaw_pred
        mask = _sequence_frame_mask(seq_mask, order=1)
    else:
        diff = _wrap_angle(dyaw_pred[:, 1:] - dyaw_pred[:, :-1])  # (B, T-2)
        mask = _sequence_frame_mask(seq_mask, order=2)

    if mask is not None and mask.sum().item() == 0:
        return pred_rotation.new_zeros(())

    if cfg.robust:
        per_step = nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    else:
        per_step = diff.pow(2)

    return _masked_mean(per_step, mask)


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
            )

    def forward(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        *,
        seq_mask: Tensor | None = None,
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

        # Position loss (optionally mask padded frames)
        if pred_position.dim() == 3 and seq_mask is not None:
            per_elem = nn.functional.smooth_l1_loss(
                pred_position, target_position, reduction="none"
            ).mean(dim=-1)  # (B, T)
            pos_loss = _masked_mean(per_elem, seq_mask)
        else:
            pos_loss = position_loss(pred_position, target_position)

        # Rotation loss (optionally mask padded frames)
        if pred_rotation.dim() == 3 and seq_mask is not None:
            pred_norm = nn.functional.normalize(pred_rotation, dim=-1)
            cos_sim = (pred_norm * target_rotation).sum(dim=-1)  # (B, T)
            per_frame = 1.0 - cos_sim
            rot_loss = _masked_mean(per_frame, seq_mask)
        else:
            rot_loss = rotation_loss(pred_rotation, target_rotation)

        total = (
            self.config.position_weight * pos_loss
            + self.config.rotation_weight * rot_loss
        )

        temp_loss = zero
        pos_temp_gt = zero
        pos_temp_inertia = zero
        rot_temp_gt = zero
        rot_temp_inertia = zero

        # Temporal consistency losses (for sequences)
        cfg = self.config.temporal
        if cfg.position_gt.weight > 0.0:
            pos_temp_gt = position_temporal_match_loss(
                pred_position, target_position, cfg.position_gt, seq_mask=seq_mask
            )
            total = total + cfg.position_gt.weight * pos_temp_gt
            temp_loss = temp_loss + pos_temp_gt

        if cfg.position_inertia.weight > 0.0:
            pos_temp_inertia = position_temporal_inertia_loss(
                pred_position, cfg.position_inertia, seq_mask=seq_mask
            )
            total = total + cfg.position_inertia.weight * pos_temp_inertia
            temp_loss = temp_loss + pos_temp_inertia

        if cfg.rotation_gt.weight > 0.0:
            rot_temp_gt = rotation_temporal_match_loss(
                pred_rotation, target_rotation, cfg.rotation_gt, seq_mask=seq_mask
            )
            total = total + cfg.rotation_gt.weight * rot_temp_gt
            temp_loss = temp_loss + rot_temp_gt

        if cfg.rotation_inertia.weight > 0.0:
            rot_temp_inertia = rotation_temporal_inertia_loss(
                pred_rotation, cfg.rotation_inertia, seq_mask=seq_mask
            )
            total = total + cfg.rotation_inertia.weight * rot_temp_inertia
            temp_loss = temp_loss + rot_temp_inertia

        return {
            "total": total,
            "position": pos_loss,
            "rotation": rot_loss,
            "temporal": temp_loss,
            "position_temporal_gt": pos_temp_gt,
            "position_temporal_inertia": pos_temp_inertia,
            "rotation_temporal_gt": rot_temp_gt,
            "rotation_temporal_inertia": rot_temp_inertia,
        }
