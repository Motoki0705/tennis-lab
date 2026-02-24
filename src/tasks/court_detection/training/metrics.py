"""Evaluation metrics for court keypoint detection."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


class CourtKeypointMetrics(Metric):
    """Metrics for court keypoint detection.

    Computes:
        - PCK (Percentage of Correct Keypoints) at various thresholds
        - Mean keypoint error in pixels
        - Visibility accuracy

    Args:
        pck_thresholds: List of PCK thresholds (fraction of image diagonal).
        image_size: Image size [H, W] for computing pixel errors.
    """

    def __init__(
        self,
        pck_thresholds: list[float] | None = None,
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        self.pck_thresholds = pck_thresholds or [0.05, 0.1, 0.2]
        self.image_size = image_size

        # Compute image diagonal for PCK normalization
        self.diagonal = (image_size[0] ** 2 + image_size[1] ** 2) ** 0.5

        # State variables
        self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "correct_per_threshold",
            default=torch.zeros(len(self.pck_thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "visibility_correct",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "visibility_total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        pred_keypoints: Tensor,
        target_keypoints: Tensor,
        pred_visibility: Tensor,
        target_visibility: Tensor,
    ) -> None:
        """Update metric state.

        Args:
            pred_keypoints: Predicted keypoints (B, K, 2) in normalized coords.
            target_keypoints: Target keypoints (B, K, 2) in normalized coords.
            pred_visibility: Predicted visibility probs (B, K).
            target_visibility: Target visibility (B, K).
        """
        # Only evaluate visible keypoints
        visible_mask = target_visibility > 0.5

        if visible_mask.sum() == 0:
            return

        # Scale to pixel coordinates
        pred_px = pred_keypoints.clone()
        pred_px[..., 0] *= self.image_size[1]
        pred_px[..., 1] *= self.image_size[0]

        target_px = target_keypoints.clone()
        target_px[..., 0] *= self.image_size[1]
        target_px[..., 1] *= self.image_size[0]

        # Compute per-keypoint Euclidean distance
        distances = torch.norm(pred_px - target_px, dim=-1)  # (B, K)

        # Only consider visible keypoints
        visible_distances = distances[visible_mask]

        # Update error sum
        self.total_error += visible_distances.sum()
        self.total_count += visible_mask.sum()

        # Update PCK counts
        for i, threshold in enumerate(self.pck_thresholds):
            threshold_px = threshold * self.diagonal
            correct = (visible_distances < threshold_px).sum()
            self.correct_per_threshold[i] += correct

        # Update visibility accuracy
        pred_vis_binary = pred_visibility > 0.5
        target_vis_binary = target_visibility > 0.5
        self.visibility_correct += (pred_vis_binary == target_vis_binary).sum()
        self.visibility_total += target_visibility.numel()

    def compute(self) -> dict[str, Tensor]:
        """Compute final metrics.

        Returns:
            Dictionary with:
                - 'mean_error': Mean keypoint error in pixels
                - 'pck@{threshold}': PCK at each threshold
                - 'visibility_acc': Visibility classification accuracy
        """
        results = {}

        # Mean error
        if self.total_count > 0:
            results["mean_error"] = self.total_error / self.total_count
        else:
            results["mean_error"] = torch.tensor(0.0)

        # PCK at each threshold
        for i, threshold in enumerate(self.pck_thresholds):
            if self.total_count > 0:
                pck = self.correct_per_threshold[i] / self.total_count
            else:
                pck = torch.tensor(0.0)
            results[f"pck@{threshold}"] = pck

        # Main PCK (use 0.1 threshold)
        if 0.1 in self.pck_thresholds:
            idx = self.pck_thresholds.index(0.1)
            results["pck"] = results[f"pck@{0.1}"]
        else:
            results["pck"] = results.get(f"pck@{self.pck_thresholds[0]}", torch.tensor(0.0))

        # Visibility accuracy
        if self.visibility_total > 0:
            results["visibility_acc"] = self.visibility_correct / self.visibility_total
        else:
            results["visibility_acc"] = torch.tensor(0.0)

        return results
