"""Evaluation metrics for supervised ball detection."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric

from src.utils.data.heatmaps import heatmaps_to_argmax


class BallDetectionMetrics(Metric):
    """Distance-based frame metrics derived from predicted heatmaps.

    Notes:
        ``target_coords`` are stored in original-image pixel space for clarity at
        the dataset boundary. Distance evaluation is also performed in original
        image pixels so that downstream users can interpret
        ``ball_distance_threshold`` directly without heatmap-scale conversion.
    """

    def __init__(
        self,
        *,
        peak_threshold: float = 0.5,
        ball_distance_threshold: float = 4.0,
    ) -> None:
        super().__init__()
        self.peak_threshold = float(peak_threshold)
        self.ball_distance_threshold = float(ball_distance_threshold)
        if self.peak_threshold < 0:
            raise ValueError("metrics.peak_threshold must be non-negative.")
        if self.ball_distance_threshold < 0:
            raise ValueError("metrics.ball_distance_threshold must be non-negative.")

        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("distance_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("distance_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_heatmaps: Tensor,
        target_coords: Tensor,
        target_visibility: Tensor,
        original_size: Tensor,
    ) -> None:
        """Update frame-level metric state.

        Args:
            pred_heatmaps: Predicted probabilities with shape (B, T, H, W).
            target_coords: Target coordinates in original image pixel space,
                shape (B, T, 2).
            target_visibility: Target visibility flags, shape (B, T).
            original_size: Original frame size in ``(width, height)`` ordering,
                shape (B, 2).
        """
        if pred_heatmaps.ndim != 4:
            raise ValueError(
                "pred_heatmaps must have shape (B, T, H, W), "
                f"got {tuple(pred_heatmaps.shape)}."
            )
        if target_coords.shape[:2] != pred_heatmaps.shape[:2]:
            raise ValueError(
                "target_coords must match pred_heatmaps batch/time shape, "
                f"got {tuple(target_coords.shape)} vs {tuple(pred_heatmaps.shape)}."
            )
        if target_visibility.shape != pred_heatmaps.shape[:2]:
            raise ValueError(
                "target_visibility must have shape (B, T), "
                f"got {tuple(target_visibility.shape)}."
            )
        if original_size.shape != (pred_heatmaps.shape[0], 2):
            raise ValueError(
                "original_size must have shape (B, 2), "
                f"got {tuple(original_size.shape)}."
            )

        batch_size = pred_heatmaps.shape[0]
        pred_coords_normalized, peak_values = heatmaps_to_argmax(pred_heatmaps)

        original_width = original_size[:, 0].view(batch_size, 1)
        original_height = original_size[:, 1].view(batch_size, 1)

        pred_coords_original = torch.empty_like(pred_coords_normalized, dtype=target_coords.dtype)
        pred_coords_original[..., 0] = pred_coords_normalized[..., 0].to(target_coords.dtype) * (
            torch.clamp(original_width - 1.0, min=0.0)
        )
        pred_coords_original[..., 1] = pred_coords_normalized[..., 1].to(target_coords.dtype) * (
            torch.clamp(original_height - 1.0, min=0.0)
        )

        distances_original = torch.norm(pred_coords_original - target_coords, dim=-1)
        pred_visible = peak_values > self.peak_threshold
        target_visible = target_visibility > 0.5
        matched = pred_visible & target_visible & (distances_original < self.ball_distance_threshold)

        self.tp += matched.sum().to(torch.float32)
        self.distance_sum += distances_original[matched].sum()
        self.distance_count += matched.sum().to(torch.float32)

        false_positive = pred_visible & ~matched
        false_negative = (~pred_visible) & target_visible
        self.fp += false_positive.sum().to(torch.float32)
        self.fn += false_negative.sum().to(torch.float32)

    def compute(self) -> dict[str, Tensor]:
        """Compute precision, recall, F1, and matched mean distance."""
        precision = self.tp / torch.clamp(self.tp + self.fp, min=1.0)
        recall = self.tp / torch.clamp(self.tp + self.fn, min=1.0)
        f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        mean_distance = self.distance_sum / torch.clamp(self.distance_count, min=1.0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_distance_px": mean_distance,
        }


__all__ = ["BallDetectionMetrics"]
