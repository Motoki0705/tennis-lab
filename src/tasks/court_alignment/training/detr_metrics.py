"""Multi-instance metrics for DINO oriented-court detections."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import cast

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.tasks.court_alignment.geometry.oriented_box import oriented_box_corners
from src.tasks.court_alignment.inference.detr_decoder import CourtDetrDetections


def _image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, bool):
        raise TypeError("image_size must be a positive integer or (height,width).")
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return image_size, image_size
    if len(image_size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in image_size
    ):
        raise ValueError("image_size must contain two positive integers.")
    return image_size


def _target_geometry(
    target: Mapping[str, Tensor],
    *,
    image_size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    court_boxes = target.get("court_boxes")
    if not isinstance(court_boxes, Tensor) or court_boxes.ndim != 2 or court_boxes.shape[-1] != 5:
        raise ValueError("Every target must contain court_boxes with shape [N,5].")
    height, width = image_size
    centers = court_boxes[:, :2] * court_boxes.new_tensor((width, height))
    long_sides = court_boxes[:, 2] * float(max(height, width))
    axes = court_boxes[:, 3:]
    corners = oriented_box_corners(centers, long_sides, axes)
    return centers, long_sides, axes, corners


_CORNER_PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _pairwise_corner_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Return minimum unordered four-corner mean error for every box pair."""

    if predicted.ndim != 3 or predicted.shape[1:] != (4, 2):
        raise ValueError("predicted corners must have shape [N,4,2].")
    if target.ndim != 3 or target.shape[1:] != (4, 2):
        raise ValueError("target corners must have shape [M,4,2].")
    if predicted.device != target.device or predicted.dtype != target.dtype:
        raise ValueError("Predicted and target corners must share device and dtype.")
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        return predicted.new_empty((predicted.shape[0], target.shape[0]))
    permutation = torch.tensor(
        _CORNER_PERMUTATIONS,
        dtype=torch.long,
        device=target.device,
    )
    permuted_target = target[:, permutation]
    delta = predicted[:, None, None] - permuted_target[None]
    errors = torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)
    return cast(Tensor, errors.amin(dim=-1))


class CourtDetrMetrics:
    """Accumulate count and matched OBB pose errors across a split.

    Hungarian assignment uses the permutation-invariant mean corner error.
    Assigned pairs above ``match_max_corner_error_px`` remain explicit false
    positives and false negatives rather than being reported as detections.
    """

    def __init__(self, *, match_max_corner_error_px: float = 16.0) -> None:
        threshold = float(match_max_corner_error_px)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("match_max_corner_error_px must be finite and positive.")
        self.match_max_corner_error_px = threshold
        self.reset()

    def reset(self) -> None:
        self._sample_count = 0
        self._count_exact = 0
        self._count_absolute_error = 0.0
        self._prediction_count = 0
        self._target_count = 0
        self._true_positive = 0
        self._center_error_sum = 0.0
        self._scale_relative_error_sum = 0.0
        self._axial_angle_error_sum_deg = 0.0
        self._corner_error_sum = 0.0
        self._matched_count = 0
        self._image_diagonal_sum = 0.0

    def update(
        self,
        predictions: CourtDetrDetections,
        targets: Sequence[Mapping[str, Tensor]],
        *,
        image_size: int | tuple[int, int],
    ) -> None:
        with torch.no_grad():
            self._update(predictions, targets, image_size=image_size)

    def _update(
        self,
        predictions: CourtDetrDetections,
        targets: Sequence[Mapping[str, Tensor]],
        *,
        image_size: int | tuple[int, int],
    ) -> None:
        height, width = _image_size(image_size)
        if len(predictions) != len(targets):
            raise ValueError("Prediction and target batch sizes must agree.")
        diagonal = math.hypot(width - 1.0, height - 1.0)
        for predicted, target in zip(predictions.samples, targets, strict=True):
            target_centers, target_long_sides, target_axes, target_corners = (
                _target_geometry(target, image_size=(height, width))
            )
            prediction_count = predicted.num_instances
            target_count = int(target_centers.shape[0])
            self._sample_count += 1
            self._prediction_count += prediction_count
            self._target_count += target_count
            self._count_exact += int(prediction_count == target_count)
            self._count_absolute_error += abs(prediction_count - target_count)
            self._image_diagonal_sum += diagonal
            if prediction_count == 0 or target_count == 0:
                continue

            corner_cost = _pairwise_corner_error(
                predicted.corners_px,
                target_corners,
            )
            prediction_indices, target_indices = linear_sum_assignment(
                corner_cost.detach().float().cpu().numpy()
            )
            prediction_indices_tensor = torch.as_tensor(
                prediction_indices,
                dtype=torch.long,
                device=corner_cost.device,
            )
            target_indices_tensor = torch.as_tensor(
                target_indices,
                dtype=torch.long,
                device=corner_cost.device,
            )
            accepted = (
                corner_cost[prediction_indices_tensor, target_indices_tensor]
                <= self.match_max_corner_error_px
            )
            prediction_indices_tensor = prediction_indices_tensor[accepted]
            target_indices_tensor = target_indices_tensor[accepted]
            matched_count = int(prediction_indices_tensor.numel())
            if matched_count == 0:
                continue

            predicted_centers = predicted.centers_px[prediction_indices_tensor]
            matched_target_centers = target_centers[target_indices_tensor]
            center_error = torch.linalg.vector_norm(
                predicted_centers - matched_target_centers,
                dim=-1,
            )
            predicted_long_sides = predicted.long_sides_px[prediction_indices_tensor]
            matched_target_long_sides = target_long_sides[target_indices_tensor]
            scale_relative_error = (
                (predicted_long_sides - matched_target_long_sides).abs()
                / matched_target_long_sides.clamp_min(torch.finfo(target_long_sides.dtype).eps)
            )
            predicted_axes = predicted.axial_vectors[prediction_indices_tensor]
            matched_target_axes = target_axes[target_indices_tensor]
            double_angle = (
                predicted_axes * matched_target_axes
            ).sum(dim=-1).clamp(-1.0, 1.0)
            axial_angle_error_deg = 0.5 * torch.rad2deg(torch.acos(double_angle))
            corner_error = corner_cost[
                prediction_indices_tensor,
                target_indices_tensor,
            ]

            self._true_positive += matched_count
            self._matched_count += matched_count
            self._center_error_sum += float(center_error.sum())
            self._scale_relative_error_sum += float(scale_relative_error.sum())
            self._axial_angle_error_sum_deg += float(axial_angle_error_deg.sum())
            self._corner_error_sum += float(corner_error.sum())

    def compute(self) -> dict[str, float]:
        false_positive = self._prediction_count - self._true_positive
        false_negative = self._target_count - self._true_positive
        precision = (
            self._true_positive / self._prediction_count
            if self._prediction_count > 0
            else 0.0
        )
        recall = (
            self._true_positive / self._target_count
            if self._target_count > 0
            else 0.0
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        sample_denominator = max(self._sample_count, 1)
        if self._matched_count > 0:
            center_error = self._center_error_sum / self._matched_count
            scale_error = self._scale_relative_error_sum / self._matched_count
            angle_error = self._axial_angle_error_sum_deg / self._matched_count
            corner_error = self._corner_error_sum / self._matched_count
        else:
            center_error = self._image_diagonal_sum / sample_denominator
            scale_error = 1.0
            angle_error = 90.0
            corner_error = self._image_diagonal_sum / sample_denominator
        return {
            "instance_tp": float(self._true_positive),
            "instance_fp": float(false_positive),
            "instance_fn": float(false_negative),
            "instance_precision": float(precision),
            "instance_recall": float(recall),
            "instance_f1": float(f1),
            "instance_count_accuracy": self._count_exact / sample_denominator,
            "instance_count_mae": self._count_absolute_error / sample_denominator,
            "predicted_instance_count": float(self._prediction_count),
            "target_instance_count": float(self._target_count),
            "matched_instance_count": float(self._matched_count),
            "matched_center_mean_error_px": float(center_error),
            "matched_scale_relative_error": float(scale_error),
            "matched_axial_angle_mean_error_deg": float(angle_error),
            "matched_corner_mean_error_px": float(corner_error),
        }


__all__ = ["CourtDetrMetrics"]
