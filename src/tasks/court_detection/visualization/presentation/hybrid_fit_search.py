"""Original-pixel GT errors and explicit candidate selection for hybrid stories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.inference.contracts import CourtPrediction


@dataclass(frozen=True)
class FitErrors:
    kp_only_error: float | None
    hybrid_error: float | None
    status: str

    @property
    def improvement_px(self) -> float:
        if self.kp_only_error is None or self.hybrid_error is None:
            raise ValueError("Improvement requires both measured errors")
        return self.kp_only_error - self.hybrid_error


@dataclass(frozen=True)
class HybridFitSearch:
    min_kp_only_error_px: float
    max_hybrid_error_px: float
    max_samples: int
    top_k: int

    def __post_init__(self) -> None:
        if any(
            not np.isfinite(value) or value < 0
            for value in (self.min_kp_only_error_px, self.max_hybrid_error_px)
        ):
            raise ValueError("Search error thresholds must be finite and nonnegative")
        if self.max_samples < 1 or self.top_k < 1:
            raise ValueError("Search max_samples and top_k must be positive")

    def accepts(self, errors: FitErrors) -> bool:
        return (
            errors.status == "ok"
            and errors.kp_only_error is not None
            and errors.hybrid_error is not None
            and np.isfinite(errors.kp_only_error)
            and np.isfinite(errors.hybrid_error)
            and errors.kp_only_error >= self.min_kp_only_error_px
            and errors.hybrid_error <= self.max_hybrid_error_px
        )


def measure_errors(
    prediction: CourtPrediction,
    gt_xy: NDArray[np.floating],
    gt_visible: NDArray[np.bool_],
) -> FitErrors:
    """Average Euclidean distance on the same visible GT14 points for both fits."""
    points = np.asarray(gt_xy, dtype=np.float64)
    if (
        points.shape != (14, 2)
        or gt_visible.shape != (14,)
        or gt_visible.dtype != np.bool_
        or not gt_visible.any()
        or not np.isfinite(points[gt_visible]).all()
    ):
        raise ValueError("Error measurement requires finite visible GT KP14")
    result = prediction.homography
    if result is None:
        raise ValueError("Error measurement requires explicit hybrid postprocessing")

    def error(projected: NDArray[np.float64]) -> float:
        if projected.shape != (14, 2) or not np.isfinite(projected[gt_visible]).all():
            raise ValueError("Existing homography has invalid visible projections")
        return float(
            np.linalg.norm(projected[gt_visible] - points[gt_visible], axis=1).mean()
        )

    return FitErrors(
        error(result.kp_only.projected) if result.kp_only.matrix is not None else None,
        error(result.best.projected)
        if result.status == "ok" and result.best is not None
        else None,
        result.status,
    )
