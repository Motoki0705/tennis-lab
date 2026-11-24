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


def test_duplicate_box_prefers_high_score_in_both_input_orders() -> None:
    key = FrameKey("video/clip", "cam0", 7)
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    reference = ReferenceFrame(
        np.array([10]), boxes, np.full((1, 17, 2), 4.0), np.ones((1, 17), bool)
    )
    duplicate_boxes = np.repeat(boxes, 2, axis=0)
    scores = np.array([0.9, 0.2])
    poses: np.ndarray = np.zeros((2, 17, 3))
    poses[..., :2] = 4.0
    poses[0, :, 2] = 1.0
    first = score_players(
        {key: reference}, {key: Predictions(duplicate_boxes, scores, poses)}
    )
    reversed_rows = Predictions(duplicate_boxes[::-1], scores[::-1], poses[::-1])
    assert score_players({key: reference}, {key: reversed_rows}) == first
    detection = first["overall"]["detection"]
    assert detection["recall_by_iou"] == {"0.50": 1.0, "0.75": 1.0}
    assert all(
        recall == detection["recall_by_iou"]
        for recall in detection["recall_by_min_score"].values()
    )
    covered = first["overall"]["pose"]["with_detection_coverage"]
    assert covered["oks"]["mean"] == 1.0
    assert covered["pck"] == {"0.1": 1.0, "0.2": 1.0}


def test_unmatched_reference_keeps_the_higher_scored_duplicate_available() -> None:
    key = FrameKey("video/clip", "cam0", 7)
    boxes = np.array([[50.0, 0.0, 60.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    reference = ReferenceFrame(
        np.array([10, 11]), boxes, np.full((2, 17, 2), 4.0), np.ones((2, 17), bool)
    )
    duplicate_boxes = np.repeat(boxes[1:], 2, axis=0)
    poses: np.ndarray = np.zeros((2, 17, 3))
    poses[..., :2] = 4.0
    poses[0, :, 2] = 1.0
    candidate = Predictions(duplicate_boxes, np.array([0.9, 0.8]), poses)
    result = score_players({key: reference}, {key: candidate})
    players = result["by_player"]["video/clip"]
    assert players["10"]["pose"]["with_detection_coverage"]["oks"]["mean"] == 0.0
    assert players["11"]["pose"]["with_detection_coverage"]["oks"]["mean"] == 1.0
    assert players["11"]["detection"]["recall_by_min_score"]["0.75"] == {
        "0.50": 1.0,
        "0.75": 1.0,
    }


def test_equal_score_and_iou_use_box_then_keypoint_order() -> None:
    key = FrameKey("video/clip", "cam0", 7)
    reference = ReferenceFrame(
        np.array([10]),
        np.array([[0.0, 0.0, 10.0, 10.0]]),
        np.full((1, 17, 2), 4.0),
        np.ones((1, 17), bool),
    )
    boxes = np.array(
        [
            [1.0, 0.0, 11.0, 10.0],
            [-1.0, 0.0, 9.0, 10.0],
            [-1.0, 0.0, 9.0, 10.0],
            [-1.0, 0.0, 9.0, 10.0],
        ]
    )
    scores = np.full(4, 0.9)
    # The right-shifted box has the lowest pose x. Among left-shifted boxes,
    # prefer x=4 over x=8 and NaN.
    poses = np.broadcast_to(
        [[3.0, 4.0, 1.0], [8.0, 4.0, 1.0], [np.nan, 4.0, 1.0], [4.0, 4.0, 1.0]],
        (17, 4, 3),
    ).transpose(1, 0, 2)
    result = score_players({key: reference}, {key: Predictions(boxes, scores, poses)})
    assert (
        score_players(
            {key: reference}, {key: Predictions(boxes[::-1], scores, poses[::-1])}
        )
        == result
    )
    assert (
        result["overall"]["detection"]["xyxy_error_px"]["x_min"]["signed"]["mean"]
        == -1.0
    )
    assert (
        result["overall"]["pose"]["with_detection_coverage"]["pixel_error_px"]["mean"]
        == 0.0
    )


def test_every_metric_is_identical_under_random_candidate_permutations() -> None:
    key = FrameKey("video/clip", "cam0", 7)
    reference_boxes = np.array(
        [
            [0.0, 0.0, 12.0, 12.0],
            [6.0, 0.0, 18.0, 12.0],
            [40.0, 0.0, 60.0, 20.0],
            [80.0, 0.0, 100.0, 20.0],
        ]
    )
    joints = np.broadcast_to(reference_boxes[:, None, :2] + 4, (4, 17, 2)).copy()
    reference = ReferenceFrame(
        np.arange(4), reference_boxes, joints, np.ones((4, 17), bool)
    )
    boxes = np.array(
        [
            [-1.0, 0.0, 11.0, 12.0],
            [1.0, 0.0, 13.0, 12.0],
            [0.0, 0.0, 18.0, 12.0],
            [40.0, 0.0, 60.0, 20.0],
            [40.0, 0.0, 60.0, 20.0],
            [80.0, 0.0, 94.0, 20.0],
            [120.0, 0.0, 132.0, 12.0],
            [150.0, 0.0, 172.0, 22.0],
        ]
    )
    scores = np.array([0.6, 0.6, 0.9, 0.95, 0.95, 0.3, 0.95, 0.6])
    positions = np.array(
        [
            [5.0, 4.0, 1.0],
            [3.0, 4.0, 1.0],
            [10.0, 4.0, 1.0],
            [44.0, 4.0, 1.0],
            [np.nan, 4.0, 1.0],
            [87.0, 4.0, 1.0],
            [124.0, 4.0, 1.0],
            [154.0, 4.0, 1.0],
        ]
    )
    poses = np.broadcast_to(positions[:, None, :], (len(boxes), 17, 3))
    refs = {key: reference, FrameKey(key.clip_id, key.camera_id, 9): reference}
    expected = score_players(refs, {key: Predictions(boxes, scores, poses)})
    assert expected["overall"]["detection"]["recall_by_min_score"] == {
        "0.25": {"0.50": 0.5, "0.75": 0.25},
        "0.5": {"0.50": 0.375, "0.75": 0.25},
        "0.75": {"0.50": 0.25, "0.75": 0.125},
    }
    rng = np.random.default_rng(20)
    for _ in range(12):
        order = rng.permutation(len(boxes))
        assert (
            score_players(
                refs, {key: Predictions(boxes[order], scores[order], poses[order])}
            )
            == expected
        )
