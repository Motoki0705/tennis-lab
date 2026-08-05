"""Aggregate and source-stratified ball-detection metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.ball_detection.evaluation.contracts import MetricsSpec
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.utils.data.heatmaps import heatmaps_to_peaks


@dataclass
class _FrameCounts:
    total: int = 0
    negative: int = 0
    negative_false_positive: int = 0

    def to_dict(self) -> dict[str, int | float | None]:
        negative_fpr = (
            None if self.negative == 0 else self.negative_false_positive / self.negative
        )
        return {
            "frames": self.total,
            "negative_frames": self.negative,
            "negative_false_positive_frames": self.negative_false_positive,
            "negative_frame_fpr": negative_fpr,
        }


class StratifiedBallMetrics:
    """Compute identical metrics overall and for every dataset source."""

    def __init__(self, spec: MetricsSpec) -> None:
        self.spec = spec
        self._overall = self._new_tracker()
        self._by_source: dict[str, BallDetectionMetrics] = {}
        self._overall_counts = _FrameCounts()
        self._source_counts: dict[str, _FrameCounts] = {}
        self._device: torch.device | None = None

    def update(
        self,
        pred_heatmaps: Tensor,
        target_coords: Tensor,
        target_visibility: Tensor,
        original_size: Tensor,
        *,
        sources: list[str],
    ) -> None:
        """Update aggregate and per-source state for one sequential batch."""
        batch_size = pred_heatmaps.shape[0]
        if len(sources) != batch_size:
            raise ValueError(
                f"Expected {batch_size} source labels, got {len(sources)}."
            )
        self._ensure_device(pred_heatmaps.device)
        self._overall.update(
            pred_heatmaps,
            target_coords,
            target_visibility,
            original_size,
        )
        self._update_counts(
            self._overall_counts,
            pred_heatmaps,
            target_visibility,
        )

        for source in sorted(set(sources)):
            indices = torch.tensor(
                [index for index, value in enumerate(sources) if value == source],
                device=pred_heatmaps.device,
                dtype=torch.long,
            )
            if source not in self._by_source:
                self._by_source[source] = self._new_tracker()
            tracker = self._by_source[source]
            tracker.to(pred_heatmaps.device)
            tracker.update(
                pred_heatmaps.index_select(0, indices),
                target_coords.index_select(0, indices),
                target_visibility.index_select(0, indices),
                original_size.index_select(0, indices),
            )
            counts = self._source_counts.setdefault(source, _FrameCounts())
            self._update_counts(
                counts,
                pred_heatmaps.index_select(0, indices),
                target_visibility.index_select(0, indices),
            )

    def compute(self) -> dict[str, object]:
        """Return JSON-compatible aggregate and source metrics."""
        aggregate = _metric_values(self._overall)
        aggregate.update(self._overall_counts.to_dict())
        by_source: dict[str, dict[str, int | float | None]] = {}
        for source, tracker in sorted(self._by_source.items()):
            source_values = _metric_values(tracker)
            source_values.update(self._source_counts[source].to_dict())
            by_source[source] = source_values
        return {"aggregate": aggregate, "by_source": by_source}

    def _new_tracker(self) -> BallDetectionMetrics:
        return BallDetectionMetrics(
            peak_threshold=self.spec.peak_threshold,
            ball_distance_threshold=self.spec.ball_distance_threshold,
            nms_kernel=self.spec.nms_kernel,
            max_predictions_per_frame=self.spec.max_predictions_per_frame,
            subpixel_refine=self.spec.subpixel_refine,
        )

    def _ensure_device(self, device: torch.device) -> None:
        if self._device is None:
            self._overall.to(device)
            self._device = device
        elif self._device != device:
            raise RuntimeError(
                f"Metric device changed during evaluation: {self._device} -> {device}."
            )

    def _update_counts(
        self,
        counts: _FrameCounts,
        pred_heatmaps: Tensor,
        target_visibility: Tensor,
    ) -> None:
        _, _, pred_valid = heatmaps_to_peaks(
            pred_heatmaps,
            threshold=self.spec.peak_threshold,
            nms_kernel=self.spec.nms_kernel,
            max_peaks=self.spec.max_predictions_per_frame,
        )
        target_present = (target_visibility > 0.5).any(dim=-1)
        negative = ~target_present
        pred_present = pred_valid.any(dim=-1)
        counts.total += int(target_present.numel())
        counts.negative += int(negative.sum().item())
        counts.negative_false_positive += int((negative & pred_present).sum().item())


def _metric_values(
    tracker: BallDetectionMetrics,
) -> dict[str, int | float | None]:
    return {
        name: float(value.detach().cpu().item())
        for name, value in tracker.compute().items()
    }


__all__ = ["StratifiedBallMetrics"]
