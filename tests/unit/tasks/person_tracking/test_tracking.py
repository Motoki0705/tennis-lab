from dataclasses import replace

import numpy as np
import pytest

from src.tasks.person_tracking.botsort_pose import BotSortPose, BotSortPoseConfig
from src.tasks.person_tracking.contracts import DetectionFeatures, TrackCapacityExceeded
from src.tasks.person_tracking.methods import build_tracker


def features(frame: int, boxes: list[list[float]], embeddings: list[list[float]], *, score: float = .9) -> DetectionFeatures:
    n = len(boxes)
    return DetectionFeatures(frame, np.arange(frame * 10, frame * 10 + n, dtype=np.int64),
                             np.asarray(boxes, np.float32).reshape(n, 4), np.full(n, score, np.float32),
                             np.zeros((n, 17, 3), np.float32), np.asarray(embeddings, np.float32).reshape(n, 2), np.ones(n, bool))


def test_appearance_keeps_id_when_detection_order_changes_at_crossing() -> None:
    tracker = BotSortPose(30.)
    first = features(0, [[0, 0, 20, 40], [10, 0, 30, 40]], [[1, 0], [0, 1]])
    assert tracker.update(first).track_ids.tolist() == [1, 2]
    crossing = features(1, [[5, 0, 25, 40], [5, 0, 25, 40]], [[0, 1], [1, 0]])
    result = tracker.update(crossing)
    assert result.track_ids.tolist() == [2, 1]
    assert result.detection_rows.tolist() == [10, 11]


def test_strong_iou_cannot_override_appearance_conflict_even_in_low_score_stage() -> None:
    tracker = BotSortPose(30.)
    tracker.update(features(0, [[0, 0, 20, 40]], [[1, 0]]))
    assert len(tracker.update(features(1, [[0, 0, 20, 40]], [[0, 1]], score=.2)).track_ids) == 0
    recovered = tracker.update(features(2, [[0, 0, 20, 40]], [[1, 0]]))
    assert recovered.track_ids.tolist() == [1]


def test_gap_recovery_emits_only_observations_and_expires_after_elapsed_time() -> None:
    tracker = BotSortPose(2., BotSortPoseConfig(max_gap_s=1.))
    tracker.update(features(0, [[0, 0, 20, 40]], [[1, 0]]))
    assert len(tracker.update(features(1, [], [])).track_ids) == 0
    assert tracker.update(features(2, [[0, 0, 20, 40]], [[1, 0]])).track_ids.tolist() == [1]
    for frame in (3, 4, 5):
        tracker.update(features(frame, [], []))
    assert tracker.update(features(6, [[0, 0, 20, 40]], [[1, 0]])).track_ids.tolist() == [2]


def test_pose_breaks_equal_appearance_and_iou_tie_and_low_confidence_is_masked() -> None:
    tracker = BotSortPose(30.)
    first = features(0, [[0, 0, 20, 40], [0, 0, 20, 40]], [[1, 0], [1, 0]])
    first.poses[0, :, :] = [2, 4, 1]
    first.poses[1, :, :] = [18, 36, 1]
    tracker.update(first)
    following = features(1, [[0, 0, 20, 40], [0, 0, 20, 40]], [[1, 0], [1, 0]])
    following.poses[:] = first.poses[::-1]
    # One extremely wrong but low-confidence joint must not alter the match.
    following.poses[:, 0, :] = [1e6, -1e6, .01]
    assert tracker.update(following).track_ids.tolist() == [2, 1]


def test_cumulative_cap_never_recycles_expired_id() -> None:
    tracker = BotSortPose(1., BotSortPoseConfig(max_tracks=1))
    tracker.update(features(0, [[0, 0, 20, 40]], [[1, 0]]))
    tracker.update(features(1, [], []))
    tracker.update(features(2, [], []))
    with pytest.raises(TrackCapacityExceeded, match='no IDs recycled'):
        tracker.update(features(3, [[0, 0, 20, 40]], [[1, 0]]))


def test_frames_rows_and_method_must_be_explicit() -> None:
    tracker = BotSortPose(30.)
    first = features(0, [[0, 0, 20, 40]], [[1, 0]])
    tracker.update(first)
    with pytest.raises(ValueError, match='every frame'):
        tracker.update(features(2, [], []))
    with pytest.raises(ValueError, match='reused'):
        tracker.update(replace(first, frame=1))
    with pytest.raises(ValueError, match='not implemented'):
        build_tracker('deep_ocsort', fps=30., config=BotSortPoseConfig())


def test_masked_appearance_is_explicit_and_malformed_features_fail() -> None:
    first = features(0, [[0, 0, 20, 40]], [[1, 0]])
    with pytest.raises(ValueError, match='masked embeddings'):
        replace(first, appearance_valid=np.zeros(1, bool))
    masked = replace(first, embeddings=np.zeros((1, 2), np.float32), appearance_valid=np.zeros(1, bool))
    tracker = BotSortPose(30.)
    assert tracker.update(masked).track_ids.tolist() == [1]
    with pytest.raises(ValueError, match='unit norm'):
        replace(first, embeddings=np.full((1, 2), .2, np.float32))
