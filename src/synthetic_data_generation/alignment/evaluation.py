"""Independent residual evaluation for alignment evidence partitions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import (
    CorrespondenceSet,
    PartitionAssessment,
    PartitionMetrics,
    PartitionThresholds,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def evaluate_partition(
    evidence: CorrespondenceSet,
    *,
    scene_from_court: RigidTransform,
    thresholds: PartitionThresholds,
) -> PartitionAssessment:
    """Evaluate one fixed transform without refitting on the partition."""
    predicted = scene_from_court.apply(evidence.points_court)
    residuals = np.linalg.norm(predicted - evidence.points_scene, axis=1)
    metrics = metrics_from_residuals(
        residuals,
        camera_ids=evidence.observed_camera_ids,
        inlier_distance_m=thresholds.inlier_distance_m,
    )
    return PartitionAssessment.evaluate(metrics, thresholds)


def metrics_from_residuals(
    residuals_m: NDArray[np.floating],
    *,
    camera_ids: tuple[str, ...],
    inlier_distance_m: float,
) -> PartitionMetrics:
    """Summarize finite non-negative residuals with deterministic gates."""
    residuals = np.asarray(residuals_m, dtype=np.float64)
    if residuals.ndim != 1 or len(residuals) == 0:
        raise ValueError("residuals_m must be a non-empty 1D array.")
    if not np.isfinite(residuals).all() or np.any(residuals < 0.0):
        raise ValueError("residuals_m must be finite and non-negative.")
    if not math.isfinite(inlier_distance_m) or inlier_distance_m <= 0.0:
        raise ValueError("inlier_distance_m must be positive and finite.")
    inlier_count = int(np.count_nonzero(residuals <= inlier_distance_m))
    return PartitionMetrics(
        camera_ids=camera_ids,
        correspondence_count=len(residuals),
        inlier_count=inlier_count,
        inlier_fraction=inlier_count / len(residuals),
        rms_error_m=float(np.sqrt(np.mean(np.square(residuals)))),
        q95_error_m=float(np.quantile(residuals, 0.95)),
        maximum_error_m=float(np.max(residuals)),
    )


__all__ = ["evaluate_partition", "metrics_from_residuals"]
