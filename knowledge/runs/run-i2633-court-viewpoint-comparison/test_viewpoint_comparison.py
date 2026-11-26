"""Small independent geometric and selection checks for the experiment rules."""

import math
from dataclasses import replace

import numpy as np
import pytest
from prepare import View, matched_selection, select_cells
from protocol import (
    CENTERED_COURT,
    COURT,
    HOMOGENEOUS_COURT,
    annotate_view,
    capture_match,
    score_prediction,
)


def test_near_requires_one_captured_camera_to_pass_all_four_limits():
    forward = np.array([1.0, 0.0, 0.0])
    centers = np.array([[0, 0, 3.0], [5.0, 0, 0.0], [1.0, 0.0, 0.5]])
    forwards = np.array([[1, 0, 0.0], [1, 0, 0.0], [0, 1.0, 0]])
    distances = capture_match(
        np.zeros(3), forward, 85.0, centers, forwards, np.array([85.0] * 3)
    )
    assert distances[0] == pytest.approx(5 / 3)
    assert distances[1] == 1
    np.testing.assert_array_equal(distances[2], [5, 0, 0, 0])

    centers[2] = [3.0, 0.0, 1.0]
    forwards[2] = [1.0, 0.0, 0.0]
    boundary = capture_match(
        np.zeros(3), forward, 85.0, centers, forwards, np.array([85.0, 85.0, 97.0])
    )
    assert boundary[0] == 1.0
    assert boundary[1] == 2
    np.testing.assert_array_equal(boundary[2], [3, 1, 0, 12])


def test_broad_dataset_keeps_fixed_anchors_and_visits_distinct_cells():
    example = View("near", "g1", "court", "j", 0.0, 1.0, 85.0, 0, 0.5, (1, 0, 0, 0), 7)
    near = [replace(example, sample_id=f"near{i}") for i in range(5)]
    far = [
        replace(example, sample_id=f"far{i}", capture_distance=2, azimuth_deg=az)
        for i, az in enumerate([50.0, 50.0, 110.0, -50.0])
    ]
    narrow, broad = matched_selection(near + far, 4)
    assert len(narrow) == len(broad) == 4
    assert set(narrow) & set(broad) == {narrow[0]}
    assert {view.azimuth_deg for view in broad if view not in narrow} == {50, 110, -50}
    assert matched_selection(list(reversed(near + far)), 4) == (narrow, broad)
    with pytest.raises(ValueError, match="Needed 6"):
        select_cells(near, 6)


def test_ground_truth_controls_low_angle_side_on_and_cropped_strata():
    low = CENTERED_COURT * [30, 3] + [250, 250]
    frame = annotate_view(
        "video_name_01", {"keypoints": low, "width": 500, "height": 500}
    )
    assert frame["elevation_proxy_deg"] == pytest.approx(
        math.degrees(math.asin(0.1)), abs=1e-6
    )
    assert set(frame["strata"]) == {"all", "low_angle", "challenging", "unclassified"}

    # H's third row has known sideline/endline coefficients, and some points
    # are outside the left border. Neither fact depends on model outputs.
    homography = np.array([[20, 0, 80], [3, 15, 250], [0.013, 0.001, 1]])
    projected = HOMOGENEOUS_COURT @ homography.T
    xy = projected[:, :2] / projected[:, 2:]
    side = annotate_view(
        "video_name_02", {"keypoints": xy, "width": 500, "height": 500}
    )
    assert side["azimuth_proxy_deg"] == pytest.approx(
        math.degrees(math.atan(13)), abs=1e-4
    )
    assert side["visible_kp"] < 14
    assert {"oblique", "side_on", "cropped", "challenging"} <= set(side["strata"])


def test_homography_failure_and_wrong_geometry_remain_in_scoring():
    gt = COURT * 8 + [250.0, 250.0]
    valid = np.ones(14, dtype=bool)
    exact = score_prediction(gt, np.ones(14), valid, gt, 1024, 500)
    assert exact["h_status"] == "ok"
    assert exact["h_at_1pct"] is True
    assert exact["kp_errors_px512"] == [0.0] * 14

    wrong = score_prediction(gt + [20, 0], np.ones(14), valid, gt, 1024, 500)
    assert wrong["h_status"] == "ok"
    assert wrong["h_at_1pct"] is False
    assert wrong["kp_errors_px512"] == [10.0] * 14

    failure = score_prediction(gt, np.zeros(14), valid, gt, 1024, 500)
    assert failure["h_status"] == "insufficient_points"
    assert failure["h_mean_error_diagonal"] is None
    assert failure["h_at_1pct"] is False
