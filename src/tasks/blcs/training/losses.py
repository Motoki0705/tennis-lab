"""Loss functions for BLCS training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.blcs.models.components.differentiable_projection import (
    DifferentiableProjection,
)
from src.utils.tensor_utils import masked_mean


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
    return masked_mean(loss, mask, eps=1e-8)


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
    projector: DifferentiableProjection,
    mask: Tensor | None = None,
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
        position_norm=pred_position,
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
    if mask is not None:
        effective_mask = effective_mask * mask.unsqueeze(1)  # broadcast (B,1,T)

    loss = nn.functional.smooth_l1_loss(pred_uv, target_uv, reduction="none")  # (B, N, T, 2)
    return masked_mean(loss, effective_mask.unsqueeze(-1).expand_as(loss), eps=1e-8)


class BLCSLoss(nn.Module):
    """Combined loss for BLCS training.

    Combines position and reprojection losses with configurable weights.
    Temporal consistency is enforced by the GAN discriminator.
    Set ``position_weight=0`` to train purely from 2D supervision.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        reprojection_weight: float = 0.0,
    ) -> None:
        """Initialize the loss module.

        Args:
            position_weight: Weight for 3D position loss.
            reprojection_weight: Weight for multi-view reprojection loss.

        """
        super().__init__()
        self.position_weight = position_weight
        self.reprojection_weight = reprojection_weight

        # Lazily created when reprojection loss is needed
        self._projector: DifferentiableProjection | None = None
        if reprojection_weight > 0:
            self._projector = DifferentiableProjection()

    @property
    def projector(self) -> DifferentiableProjection:
        """Return the differentiable projection module (create on first use)."""
        if self._projector is None:
            self._projector = DifferentiableProjection()
        return self._projector

    def forward(
        self,
        pred_position: Tensor,
        target_position: Tensor | None = None,
        mask: Tensor | None = None,
        *,
        target_uv: Tensor | None = None,
        target_vis: Tensor | None = None,
        camera_R: Tensor | None = None,
        camera_C: Tensor | None = None,
        camera_f: Tensor | None = None,
        camera_cx: Tensor | None = None,
        camera_cy: Tensor | None = None,
        camera_w: Tensor | None = None,
        camera_h: Tensor | None = None,
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
        device = pred_position.device
        zero = torch.tensor(0.0, device=device)

        # ---- 3D position loss -----------------------------------------
        if self.position_weight > 0 and target_position is not None:
            pos_loss = trajectory_position_loss(pred_position, target_position, mask)
        else:
            pos_loss = zero

        # ---- Reprojection loss ----------------------------------------
        reproj_loss = zero

        _has_cam = (
            target_uv is not None
            and target_vis is not None
            and camera_R is not None
            and camera_C is not None
            and camera_f is not None
            and camera_cx is not None
            and camera_cy is not None
            and camera_w is not None
            and camera_h is not None
        )

        if self.reprojection_weight > 0 and _has_cam:
            assert target_uv is not None  # for type-checker
            assert target_vis is not None
            assert camera_R is not None
            assert camera_C is not None
            assert camera_f is not None
            assert camera_cx is not None
            assert camera_cy is not None
            assert camera_w is not None
            assert camera_h is not None
            reproj_loss = reprojection_loss(
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

        # ---- Total ----------------------------------------------------
        total = (
            self.position_weight * pos_loss
            + self.reprojection_weight * reproj_loss
        )

        return {
            "total": total,
            "position": pos_loss,
            "reprojection": reproj_loss,
        }
