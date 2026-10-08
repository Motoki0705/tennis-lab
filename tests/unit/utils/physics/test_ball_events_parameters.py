"""Event peak picking/matching and parameter error reports."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.physics.ball.events import event_detection_report, pick_event_frames
from src.utils.physics.ball.parameters import (
    field_parameter_report,
    segment_parameter_report,
)


def test_peaks_are_local_maxima_above_threshold_and_separated() -> None:
    probability = np.array([0.1, 0.7, 0.6, 0.9, 0.2, 0.55, 0.55, 0.1, 0.4, 0.8])
    # 3 beats 1 (too close); the 0.55 plateau keeps its later frame (strict right).
    assert pick_event_frames(probability, 0.5, 3).tolist() == [3, 6, 9]
    assert pick_event_frames(probability, 0.5, 1).tolist() == [1, 3, 6, 9]
    with pytest.raises(ValueError):
        pick_event_frames(probability, 1.0, 1)


def test_matching_is_one_to_one_within_tolerance() -> None:
    report = event_detection_report(
        [np.array([10, 11, 30]), np.array([], dtype=np.int64)],
        [np.array([10, 50]), np.array([5])],
        tolerance=2,
    )
    assert (report["predicted"], report["true"], report["matched"]) == (3, 3, 1)
    assert report["precision"] == pytest.approx(1 / 3)
    assert report["recall"] == pytest.approx(1 / 3)
    assert report["f1"] == pytest.approx(1 / 3)
    assert report["mean_abs_offset_frames"] == 0.0
    assert report["segmentation_failure_rate"] == 1.0


def test_a_rally_is_segmented_only_when_all_events_match_one_to_one() -> None:
    report = event_detection_report(
        [np.array([10, 31]), np.array([4, 20]), np.array([7])],
        [np.array([11, 30]), np.array([4]), np.array([7, 15])],
        tolerance=1,
    )
    # Rally 1: spurious 20; rally 2: missed 15.
    assert report["segmentation_failure_rate"] == pytest.approx(2 / 3)
    assert report["mean_offset_frames"] == 0.0


def test_precision_is_undefined_without_predictions_but_f1_is_zero() -> None:
    report = event_detection_report(
        [np.array([], dtype=np.int64)], [np.array([4, 9])], tolerance=1
    )
    assert report["precision"] is None
    assert report["recall"] == 0.0
    assert report["f1"] == 0.0
    assert report["mean_offset_frames"] is None


def test_field_errors_are_euclidean_wind_and_relative_coefficients() -> None:
    truth = {
        "wind": np.zeros((2, 3)),
        "k_drag": np.array([0.01, 0.02]),
        "k_magnus": np.array([0.001, 0.001]),
    }
    predicted = {
        "wind": np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
        "k_drag": np.array([0.011, 0.02]),
        "k_magnus": np.array([0.0005, 0.001]),
    }
    report = field_parameter_report(predicted, truth)
    assert report["wind_error_mps"]["mean"] == pytest.approx(2.5)
    assert report["k_drag_relative_error"]["mean"] == pytest.approx(0.05)
    assert report["k_magnus_relative_error"]["median"] == pytest.approx(0.25)
    with pytest.raises(KeyError):
        field_parameter_report({"wind": truth["wind"]}, truth)


def test_spin_angle_ignores_near_zero_spins() -> None:
    zeros = np.zeros((2, 3))
    truth = {
        "position": zeros,
        "velocity": zeros,
        "spin": np.array([[100.0, 0, 0], [0.5, 0, 0]]),
        "magnus": zeros,
    }
    predicted = {**truth, "spin": np.array([[0.0, 100.0, 0], [0.0, 0.5, 0]])}
    report = segment_parameter_report(predicted, truth)
    assert report["spin_angle_error_deg"]["count"] == 1
    assert report["spin_angle_error_deg"]["mean"] == pytest.approx(90.0)
    assert report["position_error_m"]["mean"] == 0.0
    with pytest.raises(ValueError, match="Shape"):
        segment_parameter_report({**predicted, "magnus": np.zeros((1, 3))}, truth)
