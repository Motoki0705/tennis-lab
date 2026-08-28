"""Loss functions for BLCS training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.geometry.court_pose import court_position_to_world_translation
from src.utils.losses.temporal import (
    BallisticGravityPenalty,
    TemporalSmoothnessPenalty,
    ballistic_second_difference,
)
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)
from src.utils.schema.court import COURT_COORD_SCALE_Z


def _expanded_mask_mean(values: Tensor, expanded_mask: Tensor) -> Tensor:
    """Reduce values with a same-shape mask using the BLCS epsilon contract."""
    weights = expanded_mask.to(dtype=values.dtype)
    return (values * weights).sum() / (weights.sum() + 1e-8)


def trajectory_position_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    axis_weights: Tensor,
) -> Tensor:
    """Compute position loss for trajectory prediction.

    Uses Smooth L1 loss which is less sensitive to outliers.

    Args:
        pred: Predicted positions, shape (B, T, 3).
        target: Target positions, shape (B, T, 3).
        mask: Visibility mask, shape (B, T). 1 = valid, 0 = ignore.
        axis_weights: Prevalidated per-axis weights, shape (3,).

    Returns:
        Tensor: Position loss.

    """
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * axis_weights.to(device=loss.device, dtype=loss.dtype).view(1, 1, 3)
    expanded_mask = mask.unsqueeze(-1).expand_as(loss)
    return _expanded_mask_mean(loss, expanded_mask)


# ---------------------------------------------------------------------------
# Reprojection losses (2D supervision)
# ---------------------------------------------------------------------------


def reprojection_loss(
    pred_position: Tensor,
    target_uv: Tensor,
    target_vis: Tensor,
    camera_R: Tensor,
    camera_C: Tensor,
    camera_f: Tensor,
    camera_cx: Tensor,
    camera_cy: Tensor,
    camera_w: Tensor,
    camera_h: Tensor,
    projector: DifferentiablePinholeProjection,
    mask: Tensor,
) -> Tensor:
    """Compute multi-view reprojection loss.

    Projects predicted 3D positions through each camera and compares
    with observed 2D ball UV coordinates.

    Args:
        pred_position: Predicted 3D in normalised court coords, ``(B, T, 3)``.
        target_uv: Ground truth ball UV per camera, ``(B, N, T, 2)``.
        target_vis: Ball visibility per camera, ``(B, N, T)``.
        camera_R .. camera_h: Camera parameters, ``(B, N, ...)``.
        projector: Differentiable projection module.
        mask: Sequence-level padding mask, ``(B, T)``.

    Returns:
        Scalar reprojection loss.
    """
    pred_uv, in_front = projector(
        world_points=court_position_to_world_translation(pred_position),
        camera_R=camera_R,
        camera_C=camera_C,
        camera_f=camera_f,
        camera_cx=camera_cx,
        camera_cy=camera_cy,
        camera_w=camera_w,
        camera_h=camera_h,
    )  # pred_uv: (B, N, T, 2),  in_front: (B, N, T)

    # Effective mask: visible AND in front of camera AND within sequence
    effective_mask = (target_vis > 0).float() * in_front.float()  # (B, N, T)
    effective_mask = effective_mask * mask.unsqueeze(1)  # broadcast (B,1,T)

    loss = nn.functional.smooth_l1_loss(
        pred_uv, target_uv, reduction="none"
    )  # (B, N, T, 2)
    expanded_mask = effective_mask.unsqueeze(-1).expand_as(loss)
    return _expanded_mask_mean(loss, expanded_mask)


class BLCSLoss(nn.Module):
    """Combined loss for BLCS training.

    Combines position and reprojection losses with configurable weights.
    Optional physics priors (:mod:`src.utils.losses.temporal`) add the temporal
    constraint that the supervised position loss leaves unconstrained (predicted
    trajectories are otherwise 20-70x noisier in acceleration than ground truth):

    - ``smoothness_weight`` penalizes jerk (piecewise-constant acceleration),
      killing the high-frequency jitter without biasing gravity/drag.
    - ``gravity_weight`` pins the vertical curvature to ``-g``, which coupled with
      the reprojection constraint fixes the ambiguous monocular depth.

    Set ``position_weight=0`` to train purely from 2D supervision.
    """

    position_axis_weights: Tensor

    def __init__(
        self,
        position_weight: float,
        reprojection_weight: float,
        position_axis_weights: Sequence[float] | None,
        *,
        smoothness_weight: float,
        gravity_weight: float,
        smoothness_order: int,
        smoothness_beta: float,
        smoothness_axis_weights: Sequence[float] | None,
        gravity_beta: float,
        gravity: float,
        frame_dt: float,
        height_scale: float = COURT_COORD_SCALE_Z,
    ) -> None:
        """Initialize the loss module.

        Args:
            position_weight: Weight for 3D position loss.
            reprojection_weight: Weight for multi-view reprojection loss.
            position_axis_weights: Optional per-axis weights for 3D position loss.
            smoothness_weight: Weight for the jerk smoothness prior (0 = off).
            gravity_weight: Weight for the ballistic vertical-curvature prior
                (0 = off). Only meaningful for the ball (height obeys ``-g``).
            smoothness_order: Finite-difference order for the smoothness prior
                (3 = jerk).
            smoothness_beta: Smooth-L1 transition for the smoothness prior.
            smoothness_axis_weights: Optional per-axis ``(x, y, z)`` weights for
                the jerk smoothness prior. ``None`` = uniform. Down-weighting the
                height axis (e.g. ``[1, 1, 0]``) lets the gravity term own the
                vertical curvature instead of the jerk term flattening the
                ballistic arc; down-weighting an axis with legitimate sharp
                motion avoids smoothing away real direction changes.
            gravity_beta: Smooth-L1 transition for the gravity prior.
            gravity: Gravitational acceleration (m/s**2).
            frame_dt: Seconds between output frames.
            height_scale: Normalization scale of the height axis (metres).

        """
        super().__init__()
        self.position_weight = position_weight
        self.reprojection_weight = reprojection_weight
        self.smoothness_weight = smoothness_weight
        self.gravity_weight = gravity_weight
        self.position_enabled = float(position_weight > 0.0)
        self.reprojection_enabled = float(reprojection_weight > 0.0)
        self.smoothness_enabled = float(smoothness_weight > 0.0)
        self.gravity_enabled = float(gravity_weight > 0.0)
        if smoothness_axis_weights is None:
            resolved_smoothness_axis_weights = (1.0, 1.0, 1.0)
        else:
            weights = tuple(float(weight) for weight in smoothness_axis_weights)
            if len(weights) != 3:
                raise ValueError(
                    "smoothness_axis_weights must contain exactly 3 values "
                    f"for (x, y, z), got {smoothness_axis_weights}."
                )
            if any(weight < 0 for weight in weights):
                raise ValueError(
                    "smoothness_axis_weights must be non-negative, "
                    f"got {smoothness_axis_weights}."
                )
            resolved_smoothness_axis_weights = weights
        self.temporal_smoothness = TemporalSmoothnessPenalty(
            order=smoothness_order,
            beta=smoothness_beta,
            axis_weights=resolved_smoothness_axis_weights,
        )
        self.gravity_penalty = BallisticGravityPenalty(
            target_second_difference=ballistic_second_difference(
                gravity=gravity,
                dt=frame_dt,
                height_scale=height_scale,
            ),
            beta=gravity_beta,
        )
        axis_weights: tuple[float, ...]
        if position_axis_weights is None:
            axis_weights = (1.0, 1.0, 1.0)
        else:
            if len(position_axis_weights) != 3:
                raise ValueError(
                    "position_axis_weights must contain exactly 3 values "
                    f"for (x, y, z), got {position_axis_weights}."
                )
            axis_weights = tuple(float(weight) for weight in position_axis_weights)
            if any(weight < 0 for weight in axis_weights):
                raise ValueError(
                    "position_axis_weights must be non-negative, "
                    f"got {position_axis_weights}."
                )
        self.register_buffer(
            "position_axis_weights",
            torch.tensor(axis_weights, dtype=torch.float32),
            persistent=False,
        )

        self.projector = DifferentiablePinholeProjection()

    def forward(
        self,
        pred_position: Tensor,
        target_position: Tensor,
        mask: Tensor,
        *,
        target_uv: Tensor,
        target_vis: Tensor,
        camera_R: Tensor,
        camera_C: Tensor,
        camera_f: Tensor,
        camera_cx: Tensor,
        camera_cy: Tensor,
        camera_w: Tensor,
        camera_h: Tensor,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            pred_position: Predicted positions, shape (B, T, 3).
            target_position: Target 3D positions, shape (B, T, 3).
                Required when ``position_weight > 0``.
            mask: Visibility mask, shape (B, T).
            target_uv: GT ball UV per camera, ``(B, N, T, 2)``.
            target_vis: Ball visibility per camera, ``(B, N, T)``.
            camera_R .. camera_h: Camera parameters for reprojection.

        Returns:
            dict with keys ``'total'``, ``'position'``, ``'reprojection'``.

        """
        pos_loss = self.position_enabled * trajectory_position_loss(
            pred_position,
            target_position,
            mask,
            axis_weights=self.position_axis_weights,
        )
        reproj_loss = self.reprojection_enabled * reprojection_loss(
            pred_position=pred_position,
            target_uv=target_uv,
            target_vis=target_vis,
            camera_R=camera_R,
            camera_C=camera_C,
            camera_f=camera_f,
            camera_cx=camera_cx,
            camera_cy=camera_cy,
            camera_w=camera_w,
            camera_h=camera_h,
            projector=self.projector,
            mask=mask,
        )
        smooth_loss = self.smoothness_enabled * self.temporal_smoothness(
            pred_position,
            mask,
        )
        gravity_loss = self.gravity_enabled * self.gravity_penalty(
            pred_position[..., 2],
            mask,
        )

        # ---- Total ----------------------------------------------------
        total = (
            self.position_weight * pos_loss
            + self.reprojection_weight * reproj_loss
            + self.smoothness_weight * smooth_loss
            + self.gravity_weight * gravity_loss
        )

        return {
            "total": total,
            "position": pos_loss,
            "reprojection": reproj_loss,
            "smoothness": smooth_loss,
            "gravity": gravity_loss,
        }
