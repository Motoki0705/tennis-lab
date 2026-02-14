"""Evaluation metrics for PLCS."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.utils.schema.keypoint_schema import COURT_COORD_SCALE_XYZ


def _flatten_valid(valid: Tensor, values: Tensor) -> Tensor:
    return values.reshape(-1, values.shape[-1])[valid]


def _valid_from_human_mask(human_mask: Tensor | None) -> Tensor | None:
    if human_mask is None:
        return None
    if human_mask.dim() == 2:
        frame_valid = human_mask > 0
    elif human_mask.dim() == 3:
        frame_valid = (human_mask > 0).any(dim=1)
    elif human_mask.dim() == 4:
        frame_valid = (human_mask > 0).any(dim=1).any(dim=-1)
    else:
        raise ValueError(
            "human_mask must be (B,T), (B,N,T), or (B,N,T,J), "
            f"got shape {tuple(human_mask.shape)}"
        )
    return frame_valid.reshape(-1)


class PLCSMetrics:
    """Compute and track PLCS evaluation metrics.

    Metrics include:
    - Position error in meters (denormalized)
    - Angular error in degrees
    - Per-axis position errors
    - Position accuracy (within threshold)
    - Angular accuracy (within threshold)
    """

    def __init__(
        self,
        position_threshold_m: float = 0.5,
        angle_threshold_deg: float = 15.0,
    ) -> None:
        """Initialize the metrics tracker.

        Args:
            position_threshold_m: Threshold for position accuracy (meters).
            angle_threshold_deg: Threshold for angular accuracy (degrees).

        """
        self.position_threshold_m = position_threshold_m
        self.angle_threshold_deg = angle_threshold_deg
        self.reset()

        self._norm_scale_xyz = COURT_COORD_SCALE_XYZ

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._position_errors: list[Tensor] = []
        self._angular_errors: list[Tensor] = []
        self._x_errors: list[Tensor] = []
        self._y_errors: list[Tensor] = []
        self._z_errors: list[Tensor] = []

    def update(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        *,
        human_mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Update metrics with new predictions.

        Args:
            pred_position: Predicted normalized position, shape (B, 3) or (B, T, 3).
            pred_rotation: Predicted rotation (sin, cos), shape (B, 2) or (B, T, 2).
            target_position: Target normalized position, shape (B, 3) or (B, T, 3).
            target_rotation: Target rotation (sin, cos), shape (B, 2) or (B, T, 2).

        Returns:
            dict: Current batch metrics.

        """
        valid = _valid_from_human_mask(human_mask)

        # Flatten temporal dimension if present: (B, T, D) -> (B*T, D)
        if pred_position.dim() == 3:
            if valid is not None:
                if not valid.any():
                    return {
                        "position_error_m": 0.0,
                        "angular_error_deg": 0.0,
                        "x_error_m": 0.0,
                        "y_error_m": 0.0,
                        "z_error_m": 0.0,
                    }
                pred_position = _flatten_valid(valid, pred_position)
                target_position = _flatten_valid(valid, target_position)
            else:
                pred_position = pred_position.flatten(0, 1)
                target_position = target_position.flatten(0, 1)
        if pred_rotation.dim() == 3:
            if valid is not None:
                pred_rotation = _flatten_valid(valid, pred_rotation)
                target_rotation = _flatten_valid(valid, target_rotation)
            else:
                pred_rotation = pred_rotation.flatten(0, 1)
                target_rotation = target_rotation.flatten(0, 1)

        # Denormalize positions to meters
        scale = torch.tensor(
            list(self._norm_scale_xyz),
            device=pred_position.device,
            dtype=pred_position.dtype,
        )
        pred_meters = pred_position * scale
        target_meters = target_position * scale

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
        pred_angle = torch.atan2(pred_rotation[:, 0], pred_rotation[:, 1])
        target_angle = torch.atan2(target_rotation[:, 0], target_rotation[:, 1])
        angle_diff = pred_angle - target_angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angular_error_rad = angle_diff.abs()
        angular_error_deg = angular_error_rad * 180.0 / math.pi
        self._angular_errors.append(angular_error_deg.detach().cpu())

        return {
            "position_error_m": pos_error.mean().item(),
            "angular_error_deg": angular_error_deg.mean().item(),
            "x_error_m": x_error.mean().item(),
            "y_error_m": y_error.mean().item(),
            "z_error_m": z_error.mean().item(),
        }

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics.

        Returns:
            dict: Aggregated metrics over all updates.

        """
        if not self._position_errors:
            return {
                "position_error_m": 0.0,
                "angular_error_deg": 0.0,
                "x_error_m": 0.0,
                "y_error_m": 0.0,
                "z_error_m": 0.0,
                "position_accuracy": 0.0,
                "angle_accuracy": 0.0,
            }

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

        return {
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
            # Configurable accuracy
            "position_accuracy": pos_accuracy,
            "angle_accuracy": angle_accuracy,
            # Fixed threshold accuracies
            "position_accuracy_0.5m": pos_acc_0_5m,
            "position_accuracy_1m": pos_acc_1m,
            "position_accuracy_2m": pos_acc_2m,
            "angle_accuracy_10deg": angle_acc_10deg,
            "angle_accuracy_15deg": angle_acc_15deg,
            "angle_accuracy_30deg": angle_acc_30deg,
        }


class PLCSTemporalMetrics:
    """Temporal consistency metrics for PLCS sequences.

    This class measures how consistent the predicted trajectory is over time,
    by comparing frame-to-frame displacements (velocities) between prediction
    and ground truth in meters.
    """

    def __init__(self, velocity_threshold_m: float = 1.0) -> None:
        """Initialize temporal metrics tracker.

        Args:
            velocity_threshold_m: Threshold for velocity accuracy in meters
                per frame. Used for temporal_velocity_accuracy.

        """
        self._norm_scale_xyz = COURT_COORD_SCALE_XYZ
        self.velocity_threshold_m = velocity_threshold_m
        self.reset()

    def reset(self) -> None:
        """Reset accumulated temporal metrics."""
        self._velocity_errors: list[Tensor] = []

    def update(
        self,
        pred_position_seq: Tensor,
        target_position_seq: Tensor,
    ) -> dict[str, float]:
        """Update temporal metrics with new sequences.

        Args:
            pred_position_seq: Predicted normalized positions, shape (B, T, 3).
            target_position_seq: Target normalized positions, shape (B, T, 3).

        Returns:
            dict: Current batch temporal metrics.

        """
        if pred_position_seq.ndim != 3 or target_position_seq.ndim != 3:
            raise ValueError(
                "Temporal metrics expect position sequences of shape (B, T, 3)"
            )

        if pred_position_seq.size(1) < 2:
            # Not enough frames to compute temporal differences
            return {"temporal_velocity_error_m": 0.0}

        # Denormalize positions to meters
        scale = torch.tensor(
            list(self._norm_scale_xyz),
            device=pred_position_seq.device,
            dtype=pred_position_seq.dtype,
        )
        pred_meters = pred_position_seq * scale  # (B, T, 3)
        target_meters = target_position_seq * scale  # (B, T, 3)

        # Frame-to-frame displacements (velocities)
        pred_vel = pred_meters[:, 1:, :] - pred_meters[:, :-1, :]  # (B, T-1, 3)
        target_vel = target_meters[:, 1:, :] - target_meters[:, :-1, :]  # (B, T-1, 3)

        # Velocity error per step (Euclidean distance)
        vel_error = (pred_vel - target_vel).norm(dim=-1)  # (B, T-1)
        self._velocity_errors.append(vel_error.detach().cpu())

        return {
            "temporal_velocity_error_m": vel_error.mean().item(),
        }

    def compute(self) -> dict[str, float]:
        """Compute aggregated temporal metrics over all updates."""
        if not self._velocity_errors:
            return {
                "temporal_velocity_error_m": 0.0,
                "temporal_velocity_error_std_m": 0.0,
                "temporal_velocity_error_median_m": 0.0,
                "temporal_velocity_accuracy": 0.0,
            }

        vel_errors = torch.cat(self._velocity_errors)

        # Threshold-based accuracy: |v_pred - v_gt| < velocity_threshold_m
        vel_accuracy = (vel_errors <= self.velocity_threshold_m).float().mean().item()

        return {
            "temporal_velocity_error_m": vel_errors.mean().item(),
            "temporal_velocity_error_std_m": vel_errors.std().item(),
            "temporal_velocity_error_median_m": vel_errors.median().item(),
            "temporal_velocity_accuracy": vel_accuracy,
        }
