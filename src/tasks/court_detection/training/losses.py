"""Task-local Dice losses for court detection training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.model_io.contracts import CourtPoseLossKind

CourtConsistencyGradientFlow: TypeAlias = Literal[
    "both",
    "stopgrad_pose",
    "stopgrad_dense",
]


class CourtPoseLossTarget(Protocol):
    @property
    def translation_m(self) -> Tensor: ...

    @property
    def rotation(self) -> Tensor: ...

    @property
    def log_focal(self) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class CourtKeypointPoseConsistencyLoss:
    """Decomposed fixed-visibility KP/predicted-pose consistency objective."""

    coordinate: Tensor
    cheirality: Tensor
    auxiliary: Tensor
    mean_distance_px: Tensor
    visible_point_count: Tensor
    invalid_depth_fraction: Tensor


def _require_finite_tensor(value: Tensor, *, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _finite_float(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _prediction_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _consistency_compute_dtype(
    dense_points_xy: Tensor,
    pose_points_xy: Tensor,
    pose_depth_m: Tensor,
) -> torch.dtype:
    if pose_points_xy.dtype != pose_depth_m.dtype:
        raise TypeError("Court pose projection points and depth must share dtype.")
    if dense_points_xy.dtype == pose_points_xy.dtype:
        return _prediction_compute_dtype(dense_points_xy.dtype)
    if (
        dense_points_xy.dtype in {torch.float16, torch.bfloat16}
        and pose_points_xy.dtype == torch.float32
    ):
        return torch.float32
    raise TypeError(
        "Court consistency predictions must share dtype except for an AMP dense "
        "prediction paired with the float32 pose authority."
    )


def consistency_effective_weight(
    *,
    weight: float,
    warmup_fraction: float,
    progress: float,
) -> float:
    """Return the explicit zero-then-linear consistency weight schedule."""
    resolved_weight = _finite_float(weight, name="weight")
    resolved_warmup = _finite_float(warmup_fraction, name="warmup_fraction")
    resolved_progress = _finite_float(progress, name="progress")
    if resolved_weight < 0.0:
        raise ValueError("weight must be non-negative.")
    if not 0.0 <= resolved_warmup < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1).")
    if not 0.0 <= resolved_progress <= 1.0:
        raise ValueError("progress must be in [0, 1].")
    if resolved_progress <= resolved_warmup:
        return 0.0
    ramp = (resolved_progress - resolved_warmup) / (1.0 - resolved_warmup)
    return resolved_weight * ramp


def keypoint_pose_consistency_loss(
    dense_points_xy: Tensor,
    pose_points_xy: Tensor,
    pose_depth_m: Tensor,
    image_size: Tensor,
    point_visible: Tensor,
    *,
    huber_delta: float,
    min_depth_m: float,
    depth_scale_m: float,
    cheirality_weight: float,
    gradient_flow: CourtConsistencyGradientFlow,
) -> CourtKeypointPoseConsistencyLoss:
    """Compute normalized Huber coordinate and cheirality loss terms.

    ``point_visible`` is fixed GT authority. Predicted depth is never used to
    remove a point from either reduction; low and negative depths instead
    receive the explicit cheirality penalty.
    """
    if (
        dense_points_xy.shape != pose_points_xy.shape
        or dense_points_xy.ndim != 3
        or dense_points_xy.shape[1:] != (14, 2)
        or dense_points_xy.shape[0] <= 0
    ):
        raise ValueError("Dense and pose Court points must share shape (B,14,2).")
    batch_size, point_count, _ = dense_points_xy.shape
    if pose_depth_m.shape != (batch_size, point_count):
        raise ValueError("Predicted Court depth must have shape (B,14).")
    if point_visible.shape != (batch_size, point_count):
        raise ValueError("Court point visibility must have shape (B,14).")
    if point_visible.dtype != torch.bool:
        raise TypeError("Court point visibility must have boolean dtype.")
    if image_size.shape != (batch_size, 2) or image_size.dtype != torch.long:
        raise ValueError("Court image_size must be int64 (B,2) in (H,W) order.")
    tensors = (pose_points_xy, pose_depth_m, image_size, point_visible)
    if any(value.device != dense_points_xy.device for value in tensors):
        raise ValueError("Court consistency tensors must be on the same device.")
    _require_finite_tensor(dense_points_xy, name="Dense Court pixel coordinates")
    _require_finite_tensor(pose_points_xy, name="Pose Court pixel coordinates")
    _require_finite_tensor(pose_depth_m, name="Predicted Court depth")
    compute_dtype = _consistency_compute_dtype(
        dense_points_xy,
        pose_points_xy,
        pose_depth_m,
    )
    if bool(torch.any(image_size <= 0)):
        raise ValueError("Court image_size values must be positive.")

    delta = _finite_float(huber_delta, name="huber_delta")
    minimum_depth = _finite_float(min_depth_m, name="min_depth_m")
    depth_scale = _finite_float(depth_scale_m, name="depth_scale_m")
    depth_weight = _finite_float(cheirality_weight, name="cheirality_weight")
    if delta <= 0.0:
        raise ValueError("huber_delta must be positive.")
    if minimum_depth <= 0.0:
        raise ValueError("min_depth_m must be positive.")
    if depth_scale <= 0.0:
        raise ValueError("depth_scale_m must be positive.")
    if depth_weight < 0.0:
        raise ValueError("cheirality_weight must be non-negative.")
    if gradient_flow not in {"both", "stopgrad_pose", "stopgrad_dense"}:
        raise ValueError(
            "gradient_flow must be 'both', 'stopgrad_pose', or 'stopgrad_dense'."
        )

    visible_count = point_visible.sum()
    if int(visible_count) == 0:
        raise ValueError("Court consistency loss requires at least one GT-visible point.")

    with torch.autocast(device_type=dense_points_xy.device.type, enabled=False):
        dense_authority = dense_points_xy.to(dtype=compute_dtype)
        pose_authority = pose_points_xy.to(dtype=compute_dtype)
        depth_authority = pose_depth_m.to(dtype=compute_dtype)
        dense_for_loss = (
            dense_authority.detach()
            if gradient_flow == "stopgrad_dense"
            else dense_authority
        )
        if gradient_flow == "stopgrad_pose":
            pose_for_loss = pose_authority.detach()
            depth_for_loss = depth_authority.detach()
        else:
            pose_for_loss = pose_authority
            depth_for_loss = depth_authority

        distance_px = torch.linalg.vector_norm(
            dense_for_loss - pose_for_loss,
            dim=-1,
        )
        diagonal_px = torch.linalg.vector_norm(
            image_size.to(dtype=compute_dtype),
            dim=-1,
        )
        normalized_distance = distance_px / diagonal_px[:, None]
        coordinate_per_point = torch.where(
            normalized_distance <= delta,
            0.5 * normalized_distance.square(),
            delta * (normalized_distance - 0.5 * delta),
        )
        visible_weight = point_visible.to(dtype=compute_dtype)
        count = visible_count.to(dtype=compute_dtype)
        coordinate = (coordinate_per_point * visible_weight).sum() / count
        cheirality_per_point = F.softplus(
            (minimum_depth - depth_for_loss) / depth_scale
        )
        cheirality = (cheirality_per_point * visible_weight).sum() / count
        mean_distance_px = (distance_px * visible_weight).sum() / count
        invalid_depth_fraction = (
            ((depth_authority <= minimum_depth) & point_visible)
            .to(dtype=compute_dtype)
            .sum()
            / count
        )
    return CourtKeypointPoseConsistencyLoss(
        coordinate=coordinate,
        cheirality=cheirality,
        auxiliary=coordinate + depth_weight * cheirality,
        mean_distance_px=mean_distance_px,
        visible_point_count=visible_count,
        invalid_depth_fraction=invalid_depth_fraction,
    )


class DiceLoss(nn.Module):
    """Per-class Dice loss averaged over classes (for segmentation)."""

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Parameters
        ----------
        logits:
            ``[B, C, H, W]`` raw logits.
        targets:
            ``[B, H, W]`` int64 labels.
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).to(dtype=probs.dtype)

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def rotation_geodesic_radians(prediction: Tensor, target: Tensor) -> Tensor:
    """Return stable per-sample SO(3) geodesic angles in radians."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-2:] != (3, 3):
        raise ValueError("Rotation geodesic inputs must share shape (B,3,3).")
    if prediction.device != target.device:
        raise ValueError("Rotation geodesic inputs must share device.")
    if prediction.dtype != target.dtype:
        raise TypeError("Rotation geodesic inputs must share dtype.")
    compute_dtype = _prediction_compute_dtype(prediction.dtype)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_authority = prediction.to(dtype=compute_dtype)
        target_authority = target.to(dtype=compute_dtype)
        relative = prediction_authority @ target_authority.transpose(-1, -2)
        skew = torch.stack(
            (
                relative[:, 2, 1] - relative[:, 1, 2],
                relative[:, 0, 2] - relative[:, 2, 0],
                relative[:, 1, 0] - relative[:, 0, 1],
            ),
            dim=-1,
        )
        sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
        cosine = 0.5 * (
            relative[:, 0, 0]
            + relative[:, 1, 1]
            + relative[:, 2, 2]
            - 1.0
        )
        return torch.atan2(sine, cosine)


def pose_losses(
    prediction: CourtDecodedPose,
    target: CourtPoseLossTarget,
) -> dict[CourtPoseLossKind, Tensor]:
    """Compute explicit metric translation, SO(3), and log-focal losses."""
    translation_dtype = _prediction_compute_dtype(prediction.translation_m.dtype)
    rotation_dtype = _prediction_compute_dtype(prediction.rotation.dtype)
    focal_dtype = _prediction_compute_dtype(prediction.log_focal.dtype)
    prediction_translation = prediction.translation_m.to(dtype=translation_dtype)
    prediction_rotation = prediction.rotation.to(dtype=rotation_dtype)
    prediction_log_focal = prediction.log_focal.to(dtype=focal_dtype)
    target_translation = target.translation_m.to(
        device=prediction.translation_m.device,
        dtype=translation_dtype,
    )
    target_rotation = target.rotation.to(
        device=prediction.rotation.device,
        dtype=rotation_dtype,
    )
    target_log_focal = target.log_focal.to(
        device=prediction.log_focal.device,
        dtype=focal_dtype,
    )
    with torch.autocast(
        device_type=prediction.translation_m.device.type,
        enabled=False,
    ):
        return {
            "pose_translation": F.mse_loss(
                prediction_translation,
                target_translation,
            ),
            "pose_rotation": rotation_geodesic_radians(
                prediction_rotation,
                target_rotation,
            ).mean(),
            "pose_focal": F.mse_loss(
                prediction_log_focal,
                target_log_focal,
            ),
        }


__all__ = [
    "BinaryDiceLoss",
    "CourtConsistencyGradientFlow",
    "CourtKeypointPoseConsistencyLoss",
    "DiceLoss",
    "consistency_effective_weight",
    "keypoint_pose_consistency_loss",
    "pose_losses",
    "rotation_geodesic_radians",
    "query_keypoint_pose_consistency_loss",
    "query_pose_losses",
]

# Deprecated symbol aliases for old experiment notebooks; implementations are
# unified and never dispatch to a model-specific encoder.
query_keypoint_pose_consistency_loss = keypoint_pose_consistency_loss
query_pose_losses = pose_losses
