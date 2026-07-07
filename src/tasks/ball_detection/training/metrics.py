"""Evaluation metrics for supervised ball detection."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchmetrics import Metric

from src.utils.data.heatmaps import heatmaps_to_peaks, refine_peaks_log_parabolic


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
        nms_kernel: int = 9,
        max_predictions_per_frame: int = 8,
        subpixel_refine: bool = True,
    ) -> None:
        super().__init__()
        self.peak_threshold = float(peak_threshold)
        self.ball_distance_threshold = float(ball_distance_threshold)
        self.nms_kernel = int(nms_kernel)
        self.max_predictions_per_frame = int(max_predictions_per_frame)
        self.subpixel_refine = bool(subpixel_refine)
        if self.peak_threshold < 0:
            raise ValueError("metrics.peak_threshold must be non-negative.")
        if self.ball_distance_threshold < 0:
            raise ValueError("metrics.ball_distance_threshold must be non-negative.")
        if self.nms_kernel <= 0 or self.nms_kernel % 2 == 0:
            raise ValueError("metrics.nms_kernel must be a positive odd integer.")
        if self.max_predictions_per_frame <= 0:
            raise ValueError("metrics.max_predictions_per_frame must be positive.")

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
                shape (B, T, K, 2).
            target_visibility: Target visibility flags, shape (B, T, K).
            original_size: Original frame size in ``(width, height)`` ordering,
                shape (B, 2).
        """
        if pred_heatmaps.ndim != 4:
            raise ValueError(
                "pred_heatmaps must have shape (B, T, H, W), "
                f"got {tuple(pred_heatmaps.shape)}."
            )
        if (
            target_coords.ndim != 4
            or target_coords.shape[:2] != pred_heatmaps.shape[:2]
        ):
            raise ValueError(
                "target_coords must have shape (B, T, K, 2) and match "
                "pred_heatmaps batch/time dimensions, "
                f"got {tuple(target_coords.shape)} vs {tuple(pred_heatmaps.shape)}."
            )
        if target_coords.shape[-1] != 2:
            raise ValueError(
                "target_coords must end with an xy dimension of size 2, "
                f"got {tuple(target_coords.shape)}."
            )
        if target_visibility.shape != target_coords.shape[:-1]:
            raise ValueError(
                "target_visibility must match target_coords without xy, "
                f"got {tuple(target_visibility.shape)}."
            )
        if original_size.shape != (pred_heatmaps.shape[0], 2):
            raise ValueError(
                "original_size must have shape (B, 2), "
                f"got {tuple(original_size.shape)}."
            )

        pred_coords_normalized, _, pred_valid = heatmaps_to_peaks(
            pred_heatmaps,
            threshold=self.peak_threshold,
            nms_kernel=self.nms_kernel,
            max_peaks=self.max_predictions_per_frame,
        )
        if self.subpixel_refine:
            pred_coords_normalized = refine_peaks_log_parabolic(
                pred_heatmaps, pred_coords_normalized
            )
        for batch_index in range(pred_heatmaps.shape[0]):
            width = max(float(original_size[batch_index, 0].item()) - 1.0, 0.0)
            height = max(float(original_size[batch_index, 1].item()) - 1.0, 0.0)
            for frame_index in range(pred_heatmaps.shape[1]):
                frame_predictions = pred_coords_normalized[
                    batch_index,
                    frame_index,
                    pred_valid[batch_index, frame_index],
                ].to(target_coords.dtype)
                if frame_predictions.numel() > 0:
                    frame_predictions = frame_predictions.clone()
                    frame_predictions[:, 0] *= width
                    frame_predictions[:, 1] *= height
                gt_mask = target_visibility[batch_index, frame_index] > 0.5
                frame_targets = target_coords[
                    batch_index,
                    frame_index,
                    gt_mask,
                ].to(target_coords.dtype)
                self._update_frame_matches(frame_predictions, frame_targets)

    def _update_frame_matches(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> None:
        if predictions.numel() == 0:
            self.fn += float(targets.shape[0])
            return
        if targets.numel() == 0:
            self.fp += float(predictions.shape[0])
            return
        distances = torch.cdist(predictions, targets)
        prediction_indices_np, target_indices_np = linear_sum_assignment(
            distances.detach().cpu().numpy()
        )
        prediction_indices = torch.as_tensor(
            prediction_indices_np,
            device=distances.device,
        )
        target_indices = torch.as_tensor(
            target_indices_np,
            device=distances.device,
        )
        assigned = distances[prediction_indices, target_indices]
        matched = assigned < self.ball_distance_threshold
        matched_count = matched.sum().to(torch.float32)
        self.tp += matched_count
        self.fp += float(predictions.shape[0]) - matched_count
        self.fn += float(targets.shape[0]) - matched_count
        self.distance_sum += assigned[matched].sum()
        self.distance_count += matched_count

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
