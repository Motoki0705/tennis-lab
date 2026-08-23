"""Metrics for BLCS evaluation."""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)


class BLCSMetrics:
    """Metrics tracker for BLCS evaluation.

    Tracks position errors and accuracy metrics over batches.
    """

    def __init__(
        self,
        *,
        position_threshold_m: float,
        endpoint_threshold_m: float,
        normalization: CourtCoordinateNormalization | str = "v1",
    ) -> None:
        """Initialize metrics tracker.

        Args:
            position_threshold_m: Threshold for base position accuracy (meters), default 0.3m.
            endpoint_threshold_m: Threshold for base endpoint accuracy (meters), default 0.5m.

        """
        self.position_threshold_m = position_threshold_m
        self.endpoint_threshold_m = endpoint_threshold_m
        self.normalization = (
            normalization
            if isinstance(normalization, CourtCoordinateNormalization)
            else resolve_court_coordinate_normalization(normalization)
        )
        self.position_thresholds_m = (
            self.position_threshold_m,
            2.0 * self.position_threshold_m,
            4.0 * self.position_threshold_m,
        )
        self.endpoint_thresholds_m = (
            self.endpoint_threshold_m,
            2.0 * self.endpoint_threshold_m,
        )
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.total_position_error = 0.0
        self.total_x_error = 0.0
        self.total_y_error = 0.0
        self.total_z_error = 0.0
        self.total_endpoint_error = 0.0
        self.num_frames: float = 0.0
        self.num_sequences = 0
        self.num_correct_frames = [0.0 for _ in self.position_thresholds_m]
        self.num_correct_endpoints = [0.0 for _ in self.endpoint_thresholds_m]

    def update(
        self,
        pred_position: Tensor,
        target_position: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Update metrics with a batch of predictions.

        Args:
            pred_position: Predicted positions (normalized), shape (B, T, 3).
            target_position: Target positions (normalized), shape (B, T, 3).
            mask: Visibility mask, shape (B, T).

        Returns:
            dict: Current batch metrics.

        """
        batch_size, seq_len, _ = pred_position.shape

        # Denormalize to meters
        pred_m = self.normalization.denormalize_position(pred_position)
        target_m = self.normalization.denormalize_position(target_position)
        if not isinstance(pred_m, Tensor) or not isinstance(target_m, Tensor):
            raise TypeError("BLCS metric denormalization returned a non-tensor.")

        # Compute per-frame errors
        error = pred_m - target_m
        error_norm = torch.sqrt((error**2).sum(dim=-1) + 1e-8)  # (B, T)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=pred_position.device)

        # Count valid frames
        num_valid = mask.sum().item()
        self.num_frames += num_valid
        self.num_sequences += batch_size

        # Position errors
        masked_error = (error_norm * mask).sum().item()
        self.total_position_error += masked_error

        # Per-axis errors
        self.total_x_error += (error[:, :, 0].abs() * mask).sum().item()
        self.total_y_error += (error[:, :, 1].abs() * mask).sum().item()
        self.total_z_error += (error[:, :, 2].abs() * mask).sum().item()

        # Accuracy (within thresholds)
        for i, threshold in enumerate(self.position_thresholds_m):
            within = (error_norm < threshold).float()
            self.num_correct_frames[i] += (within * mask).sum().item()

        # Endpoint error (last valid frame per sequence)
        for b in range(batch_size):
            valid_indices = mask[b].nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                last_idx = valid_indices[-1]
                endpoint_error = error_norm[b, last_idx].item()
                self.total_endpoint_error += endpoint_error
                for i, threshold in enumerate(self.endpoint_thresholds_m):
                    if endpoint_error < threshold:
                        self.num_correct_endpoints[i] += 1

        # Return current batch metrics
        return {
            "position_error_m": masked_error / (num_valid + 1e-8),
            "x_error_m": (error[:, :, 0].abs() * mask).sum().item()
            / (num_valid + 1e-8),
            "y_error_m": (error[:, :, 1].abs() * mask).sum().item()
            / (num_valid + 1e-8),
            "z_error_m": (error[:, :, 2].abs() * mask).sum().item()
            / (num_valid + 1e-8),
        }

    def compute(self) -> dict[str, float]:
        """Compute aggregated metrics.

        Returns:
            dict: Aggregated metrics.

        """
        metrics: dict[str, float] = {
            "mean_position_error_m": self.total_position_error
            / (self.num_frames + 1e-8),
            "mean_x_error_m": self.total_x_error / (self.num_frames + 1e-8),
            "mean_y_error_m": self.total_y_error / (self.num_frames + 1e-8),
            "mean_z_error_m": self.total_z_error / (self.num_frames + 1e-8),
            "mean_endpoint_error_m": self.total_endpoint_error
            / (self.num_sequences + 1e-8),
        }

        def _format_threshold(value: float) -> str:
            formatted = f"{value:.3f}".rstrip("0").rstrip(".")
            return formatted.replace(".", "_")

        for i, threshold in enumerate(self.position_thresholds_m):
            key = f"position_accuracy_{_format_threshold(threshold)}m"
            metrics[key] = self.num_correct_frames[i] / (self.num_frames + 1e-8)

        for i, threshold in enumerate(self.endpoint_thresholds_m):
            key = f"endpoint_accuracy_{_format_threshold(threshold)}m"
            metrics[key] = self.num_correct_endpoints[i] / (self.num_sequences + 1e-8)

        return metrics
