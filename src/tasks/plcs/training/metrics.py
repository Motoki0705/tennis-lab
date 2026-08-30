"""Evaluation metrics for PLCS."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from src.tasks.base.evaluation import (
    PairedReferencePositionMetrics,
    compute_axis_wise_position_error,
    compute_paired_reference_position_metrics,
    stratify_metric_by_reference_view_index,
)
from src.tasks.base.generate_dataset import CourtReferenceFrameProvenance
from src.tasks.base.training.metric_logging import format_metric_threshold
from src.tasks.plcs.court_keypoint_contract import (
    headings_target_to_physical,
    normalized_points_target_to_physical,
)
from src.utils.geometry.angles import angular_error
from src.utils.schema.court_normalization import denormalize_court_position
from src.utils.schema.player import COCO_KP_NAMES, NUM_HUMAN_KP

CANONICAL_PCK_THRESHOLD_M = 0.1
CANONICAL_POSE_HEADLINE_KEYS = (
    "canonical_mpjpe_m",
    "canonical_pck_0.1m",
)
CANONICAL_POSE_DIAGNOSTIC_KEYS = (
    "canonical_joint_error_median_m",
    *(f"canonical_joint_error_{name}_m" for name in COCO_KP_NAMES),
)


def _flatten_valid(valid: Tensor, values: Tensor) -> Tensor:
    return values.reshape(-1, values.shape[-1])[valid]


def _physical_metric_values(
    position: Tensor,
    rotation: Tensor,
    provenance: Sequence[CourtReferenceFrameProvenance],
) -> tuple[Tensor, Tensor]:
    """Restore one model batch to the physical court metric frame."""
    records = tuple(provenance)
    if not records:
        raise ValueError("PLCS metric provenance must not be empty.")
    if len(records) == 1:
        return (
            normalized_points_target_to_physical(
                position,
                records[0],
            ),
            headings_target_to_physical(rotation, records[0]),
        )
    if position.ndim < 2 or position.shape[0] != len(records):
        raise ValueError(
            "PLCS metric batch and Court provenance cardinality do not match."
        )
    return (
        torch.stack(
            [
                normalized_points_target_to_physical(
                    position[index],
                    record,
                )
                for index, record in enumerate(records)
            ]
        ),
        torch.stack(
            [
                headings_target_to_physical(rotation[index], record)
                for index, record in enumerate(records)
            ]
        ),
    )


@dataclass(frozen=True, slots=True)
class PLCSReferenceMetricEvidence:
    """Target-frame PLCS training metric evidence."""

    position: PairedReferencePositionMetrics

    def to_flat_dict(self) -> dict[str, float]:
        """Return stable scalar names, including local-reference strata."""
        axis = self.position.axis_wise_position_error
        result = {
            "y_sign_accuracy": self.position.y_sign_accuracy,
            "x_error_m": axis.x,
            "y_error_m": axis.y,
            "z_error_m": axis.z,
        }
        result.update(
            {
                f"reference_index_{index}_position_error_m": value
                for index, value in self.position.local_reference_index_error.items()
            }
        )
        return result


def compute_plcs_reference_metric_evidence(
    prediction_position_m: Tensor,
    target_position_m: Tensor,
    reference_view_index: Tensor,
    *,
    valid_mask: Tensor | None = None,
    y_zero_tolerance_m: float = 0.0,
) -> PLCSReferenceMetricEvidence:
    """Compute PLCS position, Y-sign, axis, and local-index evidence."""
    return PLCSReferenceMetricEvidence(
        position=compute_paired_reference_position_metrics(
            prediction_position_m,
            target_position_m,
            reference_view_index,
            valid_mask=valid_mask,
            zero_tolerance=y_zero_tolerance_m,
        ),
    )


class PLCSMetrics:
    """Compute and track PLCS evaluation metrics.

    Metrics include:
    - Position error in meters (denormalized)
    - Angular error in degrees
    - Per-axis position errors
    - Position accuracy (within threshold)
    - Angular accuracy (within threshold)
    - Optional canonical-pose MPJPE, PCK, median, and fixed COCO-17 joint errors
    """

    def __init__(
        self,
        *,
        position_threshold_m: float,
        angle_threshold_deg: float,
        predict_canonical_pose: bool = False,
    ) -> None:
        """Initialize the metrics tracker.

        Args:
            position_threshold_m: Threshold for position accuracy (meters).
            angle_threshold_deg: Threshold for angular accuracy (degrees).
            predict_canonical_pose: Whether canonical-pose metrics are required.

        """
        self.position_threshold_m = position_threshold_m
        self.angle_threshold_deg = angle_threshold_deg
        self.predict_canonical_pose = predict_canonical_pose
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._position_errors: list[Tensor] = []
        self._angular_errors: list[Tensor] = []
        self._x_errors: list[Tensor] = []
        self._y_errors: list[Tensor] = []
        self._z_errors: list[Tensor] = []
        self._reference_index_error_sums: dict[int, float] = {}
        self._reference_index_error_counts: dict[int, int] = {}
        self._num_y_sign_correct = 0
        self._num_y_sign_targets = 0
        self._canonical_error_sum = 0.0
        self._canonical_error_count = 0
        self._canonical_pck_correct = 0
        self._canonical_joint_error_sums = [0.0] * NUM_HUMAN_KP
        self._canonical_joint_error_counts = [0] * NUM_HUMAN_KP
        self._canonical_errors_for_median: list[Tensor] = []

    def _update_canonical_pose_metrics(
        self,
        pred_canonical_pose: Tensor | None,
        target_canonical_pose: Tensor | None,
        *,
        expected_frame_shape: torch.Size,
        frame_valid: Tensor | None,
    ) -> dict[str, float]:
        """Validate and accumulate optional canonical-pose metric observations."""
        pose_args_present = (
            pred_canonical_pose is not None,
            target_canonical_pose is not None,
        )
        if pose_args_present[0] != pose_args_present[1]:
            raise ValueError(
                "PLCS canonical pose metrics require pred_canonical_pose and "
                "target_canonical_pose together."
            )
        if not self.predict_canonical_pose:
            if any(pose_args_present):
                raise ValueError(
                    "PLCS canonical pose metric inputs were provided while "
                    "predict_canonical_pose=False."
                )
            return {}
        if not all(pose_args_present):
            raise ValueError(
                "PLCS canonical pose metrics require pred_canonical_pose and "
                "target_canonical_pose when predict_canonical_pose=True."
            )

        pred_pose = cast(Tensor, pred_canonical_pose)
        target_pose = cast(Tensor, target_canonical_pose)
        if pred_pose.shape != target_pose.shape:
            raise ValueError(
                "PLCS canonical pose prediction and target shapes must match, got "
                f"{tuple(pred_pose.shape)} and {tuple(target_pose.shape)}."
            )
        if pred_pose.ndim not in {3, 4} or pred_pose.shape[-2:] != (
            NUM_HUMAN_KP,
            3,
        ):
            raise ValueError(
                "PLCS canonical pose metrics require shape (B, 17, 3) or "
                f"(B, T, 17, 3), got {tuple(pred_pose.shape)}."
            )
        if pred_pose.shape[:-2] != expected_frame_shape:
            raise ValueError(
                "PLCS canonical pose leading axes must match position leading axes, "
                f"got {tuple(pred_pose.shape[:-2])} and "
                f"{tuple(expected_frame_shape)}."
            )
        if pred_pose.device != target_pose.device:
            raise ValueError(
                "PLCS canonical pose prediction and target must share a device."
            )
        if frame_valid is not None and frame_valid.shape != pred_pose.shape[:-2]:
            raise ValueError(
                "PLCS canonical pose valid-frame mask must match pose leading axes, "
                f"got {tuple(frame_valid.shape)} and "
                f"{tuple(pred_pose.shape[:-2])}."
            )

        metric_dtype = (
            torch.float64
            if torch.float64 in (pred_pose.dtype, target_pose.dtype)
            else torch.float32
        )
        errors = torch.linalg.vector_norm(
            pred_pose.detach().to(dtype=metric_dtype)
            - target_pose.detach().to(dtype=metric_dtype),
            dim=-1,
        )
        valid_errors = (
            errors.reshape(-1, NUM_HUMAN_KP)
            if frame_valid is None
            else errors[frame_valid]
        )
        if valid_errors.numel() == 0:
            return {}

        error_sum = float(valid_errors.to(dtype=torch.float64).sum().cpu().item())
        observation_count = int(valid_errors.numel())
        pck_correct = int(
            (valid_errors <= valid_errors.new_tensor(CANONICAL_PCK_THRESHOLD_M))
            .sum()
            .cpu()
            .item()
        )
        joint_error_sums = (
            valid_errors.to(dtype=torch.float64).sum(dim=0).cpu().tolist()
        )
        joint_observation_count = int(valid_errors.shape[0])

        self._canonical_error_sum += error_sum
        self._canonical_error_count += observation_count
        self._canonical_pck_correct += pck_correct
        for index, joint_error_sum in enumerate(joint_error_sums):
            self._canonical_joint_error_sums[index] += float(joint_error_sum)
            self._canonical_joint_error_counts[index] += joint_observation_count
        self._canonical_errors_for_median.append(valid_errors.reshape(-1).cpu())

        metrics = {
            "canonical_mpjpe_m": error_sum / observation_count,
            "canonical_pck_0.1m": pck_correct / observation_count,
            "canonical_joint_error_median_m": float(valid_errors.median().cpu().item()),
        }
        metrics.update(
            {
                f"canonical_joint_error_{name}_m": (
                    float(joint_error_sums[index]) / joint_observation_count
                )
                for index, name in enumerate(COCO_KP_NAMES)
            }
        )
        return metrics

    def update(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        *,
        padding_mask: Tensor | None = None,
        court_reference_provenance: Sequence[CourtReferenceFrameProvenance]
        | None = None,
        reference_view_index: Tensor | None = None,
        pred_canonical_pose: Tensor | None = None,
        target_canonical_pose: Tensor | None = None,
    ) -> dict[str, float]:
        """Update metrics with new predictions.

        Args:
            pred_position: Predicted normalized position, shape (B, 3) or (B, T, 3).
            pred_rotation: Predicted rotation (cos, sin), shape (B, 2) or (B, T, 2).
            target_position: Target normalized position, shape (B, 3) or (B, T, 3).
            target_rotation: Target rotation (cos, sin), shape (B, 2) or (B, T, 2).
            pred_canonical_pose: Optional predicted canonical COCO-17 pose.
            target_canonical_pose: Optional target canonical COCO-17 pose.

        Returns:
            dict: Current batch metrics.

        """
        reference_metrics: dict[str, float] = {}
        reference_pred_position = denormalize_court_position(pred_position)
        reference_target_position = denormalize_court_position(target_position)
        if not isinstance(reference_pred_position, Tensor) or not isinstance(
            reference_target_position, Tensor
        ):
            raise TypeError("PLCS metric denormalization must preserve tensors.")
        frame_valid: Tensor | None = None
        if padding_mask is not None:
            frame_padding = (
                padding_mask.all(dim=1) if padding_mask.ndim == 3 else padding_mask
            )
            frame_valid = ~frame_padding
        canonical_metrics = self._update_canonical_pose_metrics(
            pred_canonical_pose,
            target_canonical_pose,
            expected_frame_shape=pred_position.shape[:-1],
            frame_valid=frame_valid,
        )
        if reference_view_index is not None:
            sample_valid = (
                torch.ones(
                    pred_position.shape[0],
                    dtype=torch.bool,
                    device=pred_position.device,
                )
                if frame_valid is None
                else frame_valid.reshape(frame_valid.shape[0], -1).any(dim=1)
            )
            if bool(sample_valid.any().item()):
                reference_valid = (
                    None if frame_valid is None else frame_valid[sample_valid]
                )
                valid_reference_indices = reference_view_index[sample_valid]
                valid_reference_prediction = reference_pred_position[sample_valid]
                valid_reference_target = reference_target_position[sample_valid]
                y_sign_eligible = valid_reference_target[..., 1].abs() > 0.0
                if reference_valid is not None:
                    y_sign_eligible &= reference_valid
                y_sign_correct = torch.sign(valid_reference_prediction[..., 1]).eq(
                    torch.sign(valid_reference_target[..., 1])
                )
                if bool(y_sign_eligible.any().item()):
                    reference_metrics = compute_plcs_reference_metric_evidence(
                        valid_reference_prediction,
                        valid_reference_target,
                        valid_reference_indices,
                        valid_mask=reference_valid,
                    ).to_flat_dict()
                else:
                    axis_error = compute_axis_wise_position_error(
                        valid_reference_prediction,
                        valid_reference_target,
                        valid_mask=reference_valid,
                    )
                    reference_metrics = {
                        "x_error_m": axis_error.x,
                        "y_error_m": axis_error.y,
                        "z_error_m": axis_error.z,
                    }
                self._num_y_sign_correct += int(
                    y_sign_correct[y_sign_eligible].sum().item()
                )
                self._num_y_sign_targets += int(y_sign_eligible.sum().item())
                reference_metric_dtype = (
                    torch.float64
                    if torch.float64
                    in (
                        valid_reference_prediction.dtype,
                        valid_reference_target.dtype,
                    )
                    else torch.float32
                )
                sample_errors = torch.linalg.vector_norm(
                    valid_reference_prediction.to(dtype=reference_metric_dtype)
                    - valid_reference_target.to(dtype=reference_metric_dtype),
                    dim=-1,
                ).reshape(valid_reference_indices.shape[0], -1)
                if reference_valid is not None:
                    flat_reference_valid = reference_valid.reshape(
                        valid_reference_indices.shape[0], -1
                    )
                    sample_errors = sample_errors.masked_fill(
                        ~flat_reference_valid, 0.0
                    ).sum(dim=1) / flat_reference_valid.sum(dim=1)
                else:
                    sample_errors = sample_errors.mean(dim=1)
                reference_metrics.update(
                    {
                        f"reference_index_{index}_position_error_m": value
                        for index, value in stratify_metric_by_reference_view_index(
                            sample_errors,
                            valid_reference_indices,
                        ).items()
                    }
                )
                for index, sample_error in zip(
                    valid_reference_indices.detach().cpu().tolist(),
                    sample_errors.detach().cpu().tolist(),
                    strict=True,
                ):
                    self._reference_index_error_sums[index] = (
                        self._reference_index_error_sums.get(index, 0.0)
                        + float(sample_error)
                    )
                    self._reference_index_error_counts[index] = (
                        self._reference_index_error_counts.get(index, 0) + 1
                    )

        positions_are_meters = court_reference_provenance is not None
        if court_reference_provenance is not None:
            pred_position, pred_rotation = _physical_metric_values(
                pred_position,
                pred_rotation,
                court_reference_provenance,
            )
            target_position, target_rotation = _physical_metric_values(
                target_position,
                target_rotation,
                court_reference_provenance,
            )

        valid = frame_valid.reshape(-1) if frame_valid is not None else None

        if valid is not None and not bool(valid.any().item()):
            return {}

        # Flatten temporal dimension if present: (B, T, D) -> (B*T, D).
        # Frame-profile tensors have no temporal axis, but still need their
        # padding-only samples removed.
        if pred_position.dim() == 3:
            if valid is not None:
                pred_position = _flatten_valid(valid, pred_position)
                target_position = _flatten_valid(valid, target_position)
            else:
                pred_position = pred_position.flatten(0, 1)
                target_position = target_position.flatten(0, 1)
        elif valid is not None:
            pred_position = pred_position[valid]
            target_position = target_position[valid]
        if pred_rotation.dim() == 3:
            if valid is not None:
                pred_rotation = _flatten_valid(valid, pred_rotation)
                target_rotation = _flatten_valid(valid, target_rotation)
            else:
                pred_rotation = pred_rotation.flatten(0, 1)
                target_rotation = target_rotation.flatten(0, 1)
        elif valid is not None:
            pred_rotation = pred_rotation[valid]
            target_rotation = target_rotation[valid]

        # Denormalize positions to meters
        pred_meters = (
            pred_position
            if positions_are_meters
            else denormalize_court_position(pred_position)
        )
        target_meters = (
            target_position
            if positions_are_meters
            else denormalize_court_position(target_position)
        )

        # Position error (Euclidean distance)
        pos_error = (pred_meters - target_meters).norm(dim=-1)
        self._position_errors.append(pos_error.detach().cpu())

        # Per-axis errors
        x_error = (pred_meters[:, 0] - target_meters[:, 0]).abs()
        y_error = (pred_meters[:, 1] - target_meters[:, 1]).abs()
        z_error = (pred_meters[:, 2] - target_meters[:, 2]).abs()
        self._x_errors.append(x_error.detach().cpu())
        self._y_errors.append(y_error.detach().cpu())
        self._z_errors.append(z_error.detach().cpu())

        # Angular error
        angular_error_rad = angular_error(pred_rotation, target_rotation)
        angular_error_deg = angular_error_rad * 180.0 / math.pi
        self._angular_errors.append(angular_error_deg.detach().cpu())

        return {
            "position_error_m": pos_error.mean().item(),
            "angular_error_deg": angular_error_deg.mean().item(),
            "x_error_m": x_error.mean().item(),
            "y_error_m": y_error.mean().item(),
            "z_error_m": z_error.mean().item(),
            **reference_metrics,
            **canonical_metrics,
        }

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics.

        Returns:
            dict: Aggregated metrics over all updates.

        """
        if not self._position_errors:
            raise RuntimeError(
                "PLCSMetrics.compute() requires at least one valid position; "
                "the epoch contained no metric observations."
            )

        position_accuracy_key = (
            f"position_accuracy_{format_metric_threshold(self.position_threshold_m)}m"
        )
        angle_accuracy_key = (
            f"angle_accuracy_{format_metric_threshold(self.angle_threshold_deg)}deg"
        )
        pos_errors = torch.cat(self._position_errors)
        angular_errors = torch.cat(self._angular_errors)
        x_errors = torch.cat(self._x_errors)
        y_errors = torch.cat(self._y_errors)
        z_errors = torch.cat(self._z_errors)

        # Accuracy metrics (within threshold)
        pos_accuracy = (pos_errors <= self.position_threshold_m).float().mean().item()
        angle_accuracy = (
            (angular_errors <= self.angle_threshold_deg).float().mean().item()
        )

        # Fixed threshold accuracies for comparison
        pos_acc_0_5m = (pos_errors <= 0.5).float().mean().item()
        pos_acc_1m = (pos_errors <= 1.0).float().mean().item()
        pos_acc_2m = (pos_errors <= 2.0).float().mean().item()
        angle_acc_10deg = (angular_errors <= 10.0).float().mean().item()
        angle_acc_15deg = (angular_errors <= 15.0).float().mean().item()
        angle_acc_30deg = (angular_errors <= 30.0).float().mean().item()

        metrics = {
            # Error metrics
            "position_error_m": pos_errors.mean().item(),
            "position_error_std_m": pos_errors.std().item(),
            "position_error_median_m": pos_errors.median().item(),
            "angular_error_deg": angular_errors.mean().item(),
            "angular_error_std_deg": angular_errors.std().item(),
            "angular_error_median_deg": angular_errors.median().item(),
            "x_error_m": x_errors.mean().item(),
            "y_error_m": y_errors.mean().item(),
            "z_error_m": z_errors.mean().item(),
            # Fixed threshold accuracies
            "position_accuracy_0.5m": pos_acc_0_5m,
            "position_accuracy_1m": pos_acc_1m,
            "position_accuracy_2m": pos_acc_2m,
            "angle_accuracy_10deg": angle_acc_10deg,
            "angle_accuracy_15deg": angle_acc_15deg,
            "angle_accuracy_30deg": angle_acc_30deg,
        }
        metrics[position_accuracy_key] = pos_accuracy
        metrics[angle_accuracy_key] = angle_accuracy
        if self._num_y_sign_targets:
            metrics["y_sign_accuracy"] = (
                self._num_y_sign_correct / self._num_y_sign_targets
            )
        metrics.update(
            {
                f"reference_index_{index}_position_error_m": (
                    total / self._reference_index_error_counts[index]
                )
                for index, total in self._reference_index_error_sums.items()
            }
        )
        if self.predict_canonical_pose:
            if self._canonical_error_count == 0:
                raise RuntimeError(
                    "PLCSMetrics.compute() requires at least one valid canonical "
                    "pose joint when predict_canonical_pose=True."
                )
            canonical_errors = torch.cat(self._canonical_errors_for_median)
            metrics.update(
                {
                    "canonical_mpjpe_m": (
                        self._canonical_error_sum / self._canonical_error_count
                    ),
                    "canonical_pck_0.1m": (
                        self._canonical_pck_correct / self._canonical_error_count
                    ),
                    "canonical_joint_error_median_m": float(
                        canonical_errors.median().item()
                    ),
                }
            )
            metrics.update(
                {
                    f"canonical_joint_error_{name}_m": (
                        self._canonical_joint_error_sums[index]
                        / self._canonical_joint_error_counts[index]
                    )
                    for index, name in enumerate(COCO_KP_NAMES)
                }
            )
        return metrics


__all__ = [
    "CANONICAL_PCK_THRESHOLD_M",
    "CANONICAL_POSE_DIAGNOSTIC_KEYS",
    "CANONICAL_POSE_HEADLINE_KEYS",
    "PLCSMetrics",
    "PLCSReferenceMetricEvidence",
    "compute_plcs_reference_metric_evidence",
]
