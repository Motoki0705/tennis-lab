"""Hand-computed partial detection and pose coverage contracts."""

import math

import numpy as np
import pytest

from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    ReferenceFrame,
    score_players,
)


def test_one_detection_cannot_match_two_players_and_an_unlabelled_box_has_no_fp() -> (
    None
):
    key = FrameKey("video/clip", "cam0", 7)
    ref = ReferenceFrame(
        np.array([1, 14]),
        np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float),
        np.zeros((2, 17, 2)),
        np.ones((2, 17), bool),
    )
    candidate = Predictions(
        np.array([[0, 0, 15, 10], [50, 10, 70, 70]], dtype=float),
        np.array([0.4, 0.9]),
    )
    result = score_players({key: ref}, {key: candidate})
    detection = result["overall"]["detection"]
    assert detection["recall_by_iou"] == {"0.50": 0.5, "0.75": 0}
    assert detection["miss_rate_at_iou_0_50"] == 0.5
    assert detection["recall_by_min_score"]["0.5"]["0.50"] == 0
    assert detection["recall_by_min_score"]["0.25"]["0.50"] == 0.5
    assert detection["matched_iou"]["median"] == pytest.approx(2 / 3)
    assert detection["unmatched_predictions"] == 1
    assert detection["precision"] is None and detection["average_precision"] is None
    assert (
        sum(
            p["detection"]["matched_at_iou_0_50"]
            for p in result["by_player"]["video/clip"].values()
        )
        == 1
    )
    assert result["by_size"]["small"]["detection"]["references"] == 2


def test_pose_reports_missing_person_and_missing_joint_in_coverage_denominators() -> (
    None
):
    key = FrameKey("video/clip", "cam0", 7)
    absent = FrameKey("video/clip", "cam0", 9)
    joints = np.full((1, 17, 2), 4.0)
    mask = np.arange(17)[None] < 3
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    reference = ReferenceFrame(np.array([2]), boxes, joints, mask)
    keypoints = np.zeros((1, 17, 3))
    keypoints[0, :3] = [(5.0, 4.0, 1.5), (6.0, 4.0, 0.9), (4.0, 4.0, 0.0)]
    result = score_players(
        {key: reference, absent: reference},
        {key: Predictions(boxes, np.array([1.0]), keypoints)},
    )
    assert result["overall"]["detection"]["recall_by_iou"] == {"0.50": 0.5, "0.75": 0.5}
    pose = result["overall"]["pose"]
    matched = pose["matched_only"]
    covered = pose["with_detection_coverage"]
    expected_oks = (math.exp(-1 / 0.5408) + math.exp(-4 / 0.5)) / 3
    assert matched["oks"]["mean"] == pytest.approx(expected_oks)
    assert covered["oks"]["mean"] == pytest.approx(expected_oks / 2)
    assert matched["pck"] == {"0.1": 1 / 3, "0.2": 2 / 3}
    assert covered["pck"] == {"0.1": 1 / 6, "0.2": 2 / 6}
    assert covered["eligible_joints"] == 6 and covered["predicted_joints"] == 2
    assert covered["valid_joint_fraction"] == pytest.approx(1 / 3)
    assert covered["pixel_error_px"] == pytest.approx(
        {"count": 2, "mean": 1.5, "median": 1.5, "p95": 1.95}
    )
    assert covered["by_joint"]["right_eye"]["eligible"] == 2
    assert covered["by_joint"]["right_eye"]["pixel_error_px"]["count"] == 0


def test_prediction_for_a_frame_outside_the_reference_is_rejected() -> None:
    key = FrameKey("video/clip", "cam0", 7)
    reference = ReferenceFrame(
        np.array([], dtype=int),
        np.empty((0, 4)),
        np.empty((0, 17, 2)),
        np.empty((0, 17), bool),
    )
    candidate = Predictions(np.empty((0, 4)), np.empty(0))
    with pytest.raises(ValueError, match="outside the fixed reference"):
        score_players(
            {key: reference}, {FrameKey(key.clip_id, key.camera_id, 8): candidate}
        )
