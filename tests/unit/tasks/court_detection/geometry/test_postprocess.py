"""Tests for homography-constrained court keypoint post-processing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.geometry import (
    court_template_xy,
    project_points,
    refine_court_keypoints_with_homography,
)


def _base_homography(*, tx: float = 320.0, ty: float = 210.0) -> NDArray[np.float32]:
    return np.array(
        [
            [28.0, 1.5, tx],
            [3.0, -18.0, ty],
            [0.002, -0.003, 1.0],
        ],
        dtype=np.float32,
    )


def test_homography_postprocess_rejects_outliers_and_restores_template_projection() -> None:
    template = court_template_xy()
    expected = project_points(template, _base_homography())
    observed = expected.copy()
    outlier_indices = np.array([2, 3, 5])
    observed[outlier_indices] += np.array([[80.0, -60.0], [-75.0, 55.0], [65.0, 70.0]])

    result = refine_court_keypoints_with_homography(
        observed[None, ...],
        np.ones((1, template.shape[0]), dtype=np.float32),
        min_score=0.3,
        ransac_reproj_threshold=2.0,
    )

    np.testing.assert_allclose(result.visibility, 1.0)
    np.testing.assert_allclose(result.keypoints[0], expected, atol=1.0e-3)
    frame_diag = result.diagnostics["frames"][0]
    assert frame_diag["success"] is True
    assert frame_diag["inlier_count"] == template.shape[0] - len(outlier_indices)
    inlier_mask = np.asarray(frame_diag["inlier_mask"], dtype=bool)
    assert not inlier_mask[outlier_indices].any()


def test_homography_postprocess_marks_seed_shortage_invisible() -> None:
    template = court_template_xy()
    observed = project_points(template, _base_homography())
    scores = np.full((1, template.shape[0]), 0.1, dtype=np.float32)
    scores[0, :3] = 0.95

    result = refine_court_keypoints_with_homography(
        observed[None, ...],
        scores,
        min_score=0.5,
    )

    np.testing.assert_allclose(result.keypoints[0], observed)
    np.testing.assert_allclose(result.visibility, 0.0)
    frame_diag = result.diagnostics["frames"][0]
    assert frame_diag["success"] is False
    assert frame_diag["reason"] == "insufficient_seed_points"
    assert frame_diag["seed_count"] == 3


def test_homography_postprocess_rejects_poor_inlier_template_coverage() -> None:
    template = court_template_xy()
    observed = project_points(template, _base_homography())
    scores = np.full((1, template.shape[0]), 0.1, dtype=np.float32)
    scores[0, [0, 6, 9, 11, 12, 13]] = 0.95

    result = refine_court_keypoints_with_homography(
        observed[None, ...],
        scores,
        min_score=0.5,
        ransac_reproj_threshold=3.0,
    )

    np.testing.assert_allclose(result.visibility, 0.0)
    frame_diag = result.diagnostics["frames"][0]
    assert frame_diag["success"] is False
    assert frame_diag["reason"] == "degenerate_inlier_coverage"
    assert frame_diag["inlier_count"] < 8


def test_temporal_median_filter_reduces_successful_frame_jitter() -> None:
    template = court_template_xy()
    stable = project_points(template, _base_homography())
    shifted = project_points(template, _base_homography(tx=340.0))
    sequence = np.stack([stable, stable, shifted, stable, stable], axis=0).astype(np.float32)
    scores = np.ones(sequence.shape[:2], dtype=np.float32)

    unfiltered = refine_court_keypoints_with_homography(
        sequence,
        scores,
        temporal_median_window=0,
    )
    filtered = refine_court_keypoints_with_homography(
        sequence,
        scores,
        temporal_median_window=3,
    )

    assert np.linalg.norm(unfiltered.keypoints[2, 0] - stable[0]) > 10.0
    np.testing.assert_allclose(filtered.keypoints[2], stable, atol=1.0e-3)
