"""Tests for fixed-transform court-line calibration and holdout metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.components.evaluation.court_lines import (
    CourtLineEvaluationSettings,
    court_line_distances,
    evaluate_projected_court_lines,
    holdout_gate_results,
    point_cloud_court_support,
    transform_points,
    transform_stability,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


def test_court_line_distance_uses_finite_painted_segments() -> None:
    points = np.asarray(
        [
            [-HALF_DOUBLES_WIDTH, 0.0],
            [0.0, HALF_LENGTH],
            [0.0, HALF_LENGTH + 2.0],
        ]
    )

    distances = court_line_distances(points)

    np.testing.assert_allclose(distances[:2], 0.0, atol=1.0e-12)
    assert distances[2] == pytest.approx(2.0)


def test_evaluation_excludes_an_adjacent_court_and_reports_coverage() -> None:
    x = np.linspace(-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH, 120)
    selected = np.column_stack(
        (
            x,
            np.full_like(x, HALF_LENGTH),
            np.zeros_like(x),
        )
    )
    adjacent = selected.copy()
    adjacent[:, 0] += 15.0
    points = np.concatenate((selected, adjacent))

    metrics = evaluate_projected_court_lines(
        points,
        np.ones(len(points)),
        court_from_scene=np.eye(4),
        settings=CourtLineEvaluationSettings(
            line_inlier_distance_m=0.1,
            template_sample_spacing_m=0.2,
        ),
    )

    assert metrics["court_roi_point_count"] == len(selected)
    assert metrics["weighted_inlier_fraction"] == pytest.approx(1.0)
    assert 0.0 < metrics["template_coverage_fraction"] < 1.0


def test_point_cloud_support_measures_metric_footprint_coverage() -> None:
    x, y = np.meshgrid(
        np.linspace(-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH, 23),
        np.linspace(-HALF_LENGTH, HALF_LENGTH, 49),
    )
    points = np.column_stack(
        (
            x.ravel(),
            y.ravel(),
            np.full(x.size, 0.03),
        )
    )

    metrics = point_cloud_court_support(
        points,
        court_from_scene=np.eye(4),
        settings=CourtLineEvaluationSettings(),
    )

    assert metrics["support_point_count"] == len(points)
    assert metrics["residual_rms_m"] == pytest.approx(0.03)
    assert metrics["occupied_grid_fraction"] == pytest.approx(1.0)


def test_transform_validation_rejects_reflection_and_anisotropy() -> None:
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match="reflection"):
        transform_points(np.zeros((1, 3)), reflection)

    anisotropic = np.diag((1.0, 2.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="one positive scale"):
        transform_points(np.zeros((1, 3)), anisotropic)


def test_transform_stability_resolves_180_degree_court_symmetry() -> None:
    rotated = np.diag((-1.0, -1.0, 1.0, 1.0))

    metrics = transform_stability(np.eye(4), rotated)

    assert metrics["orientation_difference_deg_mod_180"] == pytest.approx(0.0)
    assert metrics["centre_shift_m"] == pytest.approx(0.0)
    assert metrics["relative_scale_difference"] == pytest.approx(0.0)


def test_holdout_gates_reject_one_undercovered_group() -> None:
    gates = {
        "minimum_accepted_view_fraction": 0.8,
        "minimum_weighted_inlier_fraction": 0.75,
        "maximum_distance_weighted_q95_m": 0.4,
        "minimum_template_coverage_fraction": 0.75,
        "minimum_group_weighted_inlier_fraction": 0.6,
        "minimum_group_template_coverage_fraction": 0.35,
        "minimum_accepted_views_per_group": 8,
        "minimum_camera_height_m": 1.0,
        "maximum_camera_height_m": 3.5,
        "minimum_positive_camera_height_fraction": 1.0,
    }
    metrics = {
        "accepted_view_fraction": 1.0,
        "aggregate": {
            "weighted_inlier_fraction": 0.9,
            "distance_weighted_q95_m": 0.2,
            "template_coverage_fraction": 0.9,
        },
        "by_group": {
            "2": {
                "weighted_inlier_fraction": 0.8,
                "template_coverage_fraction": 0.8,
            },
            "6": {
                "weighted_inlier_fraction": 0.8,
                "template_coverage_fraction": 0.3,
            },
        },
        "accepted_view_count_by_group": {"2": 32, "6": 32},
        "camera_heights_m": {
            "minimum": 1.5,
            "maximum": 2.5,
            "positive_fraction": 1.0,
        },
    }

    results = holdout_gate_results(metrics, gates)

    assert results["every_group_template_coverage"] is False
    assert all(
        value
        for name, value in results.items()
        if name != "every_group_template_coverage"
    )
