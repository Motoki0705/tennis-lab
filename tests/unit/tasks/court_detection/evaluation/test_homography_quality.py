"""Tests for homography-based court annotation geometry checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.contracts import (
    HomographyEvaluationCriteria,
)
from src.tasks.court_detection.evaluation.homography_quality import (
    evaluate_homography_quality,
)
from src.tasks.court_detection.geometry import court_template_xy, project_points


def _perspective_keypoints(*, width: int, height: int) -> NDArray[np.float32]:
    homography = np.asarray(
        [
            [0.040, 0.002, 0.50],
            [0.002, -0.025, 0.48],
            [0.002, -0.012, 1.00],
        ],
        dtype=np.float32,
    )
    normalized = project_points(court_template_xy(), homography)
    return normalized * np.asarray([width - 1, height - 1], dtype=np.float32)


def _geometry_only_criteria(
    *,
    min_inliers: int = 12,
    require_ground_view: bool = False,
    max_opposite_edge_ratio: float = 0.95,
) -> HomographyEvaluationCriteria:
    return HomographyEvaluationCriteria(
        ransac_reproj_threshold_normalized=0.012,
        min_inliers=min_inliers,
        min_template_x_span_ratio=0.5,
        min_template_y_span_ratio=0.7,
        max_inlier_rms_normalized=0.006,
        min_visible_fraction=0.98,
        min_court_area_ratio=0.001,
        max_court_area_ratio=0.95,
        min_line_edge_support=0.0,
        line_distance_tolerance_px=3.0,
        line_evidence_max_side=900,
        require_ground_view=require_ground_view,
        max_opposite_edge_ratio=max_opposite_edge_ratio,
    )


def test_homography_quality_refits_two_annotation_outliers() -> None:
    width, height = 1280, 720
    expected = _perspective_keypoints(width=width, height=height)
    observed = expected.copy()
    observed[[5, 11]] += np.asarray([[180.0, -90.0], [-160.0, 120.0]])

    result = evaluate_homography_quality(
        observed,
        image_width=width,
        image_height=height,
        criteria=_geometry_only_criteria(min_inliers=12),
    )

    assert result.rejection_reasons == ()
    assert result.metrics["inlier_count"] == 12
    assert result.projected_keypoints_normalized is not None
    np.testing.assert_allclose(
        result.projected_keypoints_normalized,
        expected / np.asarray([width - 1, height - 1], dtype=np.float32),
        atol=1.0e-4,
    )


def test_homography_quality_rejects_collinear_annotations() -> None:
    x_coordinates = np.linspace(100.0, 1100.0, 14, dtype=np.float32)
    keypoints = np.stack([x_coordinates, np.full(14, 300.0, dtype=np.float32)], axis=1)

    result = evaluate_homography_quality(
        keypoints,
        image_width=1280,
        image_height=720,
        criteria=_geometry_only_criteria(),
    )

    assert result.geometry_valid is False
    assert result.rejection_reasons == ("ransac_failed",)


def test_homography_quality_rejects_affine_top_down_view_when_ground_is_required() -> (
    None
):
    width, height = 1280, 720
    template = court_template_xy()
    normalized = np.empty_like(template)
    normalized[:, 0] = template[:, 0] * 0.035 + 0.5
    normalized[:, 1] = template[:, 1] * 0.025 + 0.5
    keypoints = normalized * np.asarray([width - 1, height - 1], dtype=np.float32)

    result = evaluate_homography_quality(
        keypoints,
        image_width=width,
        image_height=height,
        criteria=_geometry_only_criteria(
            require_ground_view=True,
            max_opposite_edge_ratio=0.95,
        ),
    )

    assert result.geometry_valid is True
    assert "insufficient_ground_perspective" in result.rejection_reasons
