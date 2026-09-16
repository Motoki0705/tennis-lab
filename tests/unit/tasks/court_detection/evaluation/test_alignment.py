"""Geometry contracts for the court-alignment benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.alignment import (
    ALIGNMENT_REASON_DEGENERATE_POLYGON,
    ALIGNMENT_REASON_EMPTY_INTERSECTION,
    ALIGNMENT_REASON_INSUFFICIENT_POINTS,
    ALIGNMENT_REASON_RANSAC_FAILED,
    COURT_LINE_SEGMENTS,
    clip_convex_polygon_to_image,
    doubles_polygon_template,
    fit_template_homography,
    polygon_iou,
    project_points,
    sample_segment_points,
    symmetric_line_reprojection_error_px,
)
from src.tasks.court_detection.geometry.homography import court_template_xy

_TEMPLATE = court_template_xy(14).astype(np.float64)
_SCALE_H = np.array([[4.0, 0.0, 100.0], [0.0, 4.0, 60.0], [0.0, 0.0, 1.0]])


def test_projecting_points_is_exactly_affine_for_a_pure_scale() -> None:
    projected = project_points(_TEMPLATE, _SCALE_H)

    np.testing.assert_allclose(projected, _TEMPLATE * 4.0 + np.array([100.0, 60.0]))


def test_sampling_is_deterministic_and_includes_both_segment_ends() -> None:
    first = sample_segment_points(COURT_LINE_SEGMENTS, _TEMPLATE, per_segment=5)
    second = sample_segment_points(COURT_LINE_SEGMENTS, _TEMPLATE, per_segment=5)

    assert first.shape == (5 * len(COURT_LINE_SEGMENTS), 2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[0], _TEMPLATE[COURT_LINE_SEGMENTS[0][0]])
    np.testing.assert_allclose(first[4], _TEMPLATE[COURT_LINE_SEGMENTS[0][1]])


def test_perfect_keypoints_fit_with_every_point_as_an_inlier() -> None:
    fit = fit_template_homography(
        _TEMPLATE,
        project_points(_TEMPLATE, _SCALE_H),
        np.ones(14, dtype=bool),
        ransac_threshold_px=2.0,
    )

    assert fit.succeeded
    assert fit.inlier_count == 14
    assert fit.reason is None
    assert fit.homography is not None
    np.testing.assert_allclose(fit.homography, _SCALE_H, atol=1e-6)


def test_fewer_than_four_points_is_a_typed_failure_not_a_fallback() -> None:
    valid: NDArray[np.bool_] = np.zeros(14, dtype=bool)
    valid[:3] = True

    fit = fit_template_homography(
        _TEMPLATE,
        project_points(_TEMPLATE, _SCALE_H),
        valid,
        ransac_threshold_px=2.0,
    )

    assert not fit.succeeded
    assert fit.reason == ALIGNMENT_REASON_INSUFFICIENT_POINTS
    assert fit.inlier_count == 3


def test_collinear_points_are_rejected_as_degenerate() -> None:
    line = np.stack([np.linspace(0.0, 10.0, 14), np.zeros(14)], axis=1)
    image = line * 2.0 + 5.0

    fit = fit_template_homography(
        line, image, np.ones(14, dtype=bool), ransac_threshold_px=1.0
    )

    assert not fit.succeeded
    assert fit.reason == ALIGNMENT_REASON_RANSAC_FAILED


def test_a_shifted_prediction_keeps_the_fit_but_costs_reprojection_error() -> None:
    shifted = _SCALE_H.copy()
    shifted[0, 2] += 7.0
    reference = project_points(_TEMPLATE, _SCALE_H)
    prediction = project_points(_TEMPLATE, shifted)

    fit = fit_template_homography(
        _TEMPLATE, prediction, np.ones(14, dtype=bool), ransac_threshold_px=2.0
    )

    assert fit.succeeded
    assert np.linalg.norm(reference - prediction, axis=1).mean() == pytest.approx(
        7.0, abs=1e-6
    )
    lines = sample_segment_points(COURT_LINE_SEGMENTS, _TEMPLATE, per_segment=9)
    assert symmetric_line_reprojection_error_px(
        _SCALE_H, shifted, lines
    ) == pytest.approx(7.0)


def test_line_error_is_zero_for_two_identical_homographies() -> None:
    lines = sample_segment_points(COURT_LINE_SEGMENTS, _TEMPLATE, per_segment=7)

    assert symmetric_line_reprojection_error_px(_SCALE_H, _SCALE_H, lines) == 0.0


def test_a_polygon_entirely_outside_the_image_is_undefined() -> None:
    polygon = np.array([[-50.0, -50.0], [-10.0, -50.0], [-10.0, -10.0], [-50.0, -10.0]])

    clipped = clip_convex_polygon_to_image(polygon, 100, 100)
    assert clipped is not None and clipped.shape == (0, 2)
    value, reason = polygon_iou(polygon, polygon, width=100, height=100)

    assert value is None
    assert reason == ALIGNMENT_REASON_EMPTY_INTERSECTION


def test_non_convex_input_is_rejected_instead_of_scored() -> None:
    bow_tie = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])

    assert clip_convex_polygon_to_image(bow_tie, 100, 100) is None
    value, reason = polygon_iou(bow_tie, bow_tie, width=100, height=100)

    assert value is None
    assert reason == ALIGNMENT_REASON_DEGENERATE_POLYGON


def test_polygon_iou_is_one_for_identical_rectangles_and_half_for_a_half_split() -> (
    None
):
    full = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    half = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 100.0], [0.0, 100.0]])

    assert polygon_iou(full, full, width=100, height=100) == (1.0, None)
    assert polygon_iou(full, half, width=100, height=100) == (0.5, None)


def test_doubles_polygon_uses_the_four_outer_corners() -> None:
    polygon = doubles_polygon_template(_TEMPLATE)

    np.testing.assert_allclose(polygon, _TEMPLATE[[0, 1, 3, 2]])
