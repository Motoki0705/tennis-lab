from dataclasses import replace

import numpy as np
import pytest

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.deep_ocsort_pose import DeepOCSortPose
from src.tasks.person_tracking.strongsort import StrongSort, StrongSortConfig


def two_people(index: int, *, reverse: bool = False, appearance: bool = True,
               joints: int = 17) -> DetectionFeatures:
    # Identical boxes/appearance isolate pose evidence from geometry and Re-ID.
    boxes = np.tile(np.array([[10, 20, 50, 120]], np.float32), (2, 1))
    poses: np.ndarray = np.zeros((2, 17, 3), np.float32)
    poses[:, :, :2] = boxes[:, None, :2] + np.array([.25, .75])[:, None, None] * (boxes[:, None, 2:] - boxes[:, None, :2])
    poses[:, :joints, 2] = .8
    if reverse:
        poses = poses[::-1].copy()
    return DetectionFeatures(index, np.arange(2 * index, 2 * index + 2, dtype=np.int64), boxes,
                             np.full(2, .9, np.float32), poses,
                             np.tile(np.array([[1, 0] if appearance else [0, 0]], np.float32), (2, 1)),
                             np.full(2, appearance))


@pytest.mark.parametrize('appearance', [True, False], ids=['appearance-motion-stage', 'iou-stage'])
@pytest.mark.parametrize('joints,expected', [(3, [1, 2]), (4, [2, 1])])
def test_pose_resolves_ambiguous_assignment_in_both_stages(appearance: bool, joints: int, expected: list[int]) -> None:
    tracker = StrongSort(StrongSortConfig(pose_weight=.15))
    for index in range(3):
        tracker.update(two_people(index, joints=joints))
    result = tracker.update(two_people(3, reverse=True, appearance=appearance, joints=joints))
    assert result.track_ids.tolist() == expected
    assert result.detection_rows.tolist() == [6, 7]


def test_default_zero_pose_keeps_assignment_even_with_conflicting_pose() -> None:
    tracker = StrongSort()
    for index in range(3):
        tracker.update(two_people(index))
    assert tracker.update(two_people(3, reverse=True)).track_ids.tolist() == [1, 2]


def test_pose_cost_matches_deep_and_is_invariant_to_box_scale_and_translation() -> None:
    strong = StrongSort(StrongSortConfig(pose_weight=.15))
    deep = DeepOCSortPose(60.)
    original = two_people(0)
    strong.update(original)
    deep.update(original)
    current = two_people(1, reverse=True, joints=4)
    transformed = replace(current, boxes=current.boxes * 2 + 30, poses=current.poses.copy())
    transformed.poses[:, :, :2] = current.poses[:, :, :2] * 2 + 30
    rows = np.array([0, 1], np.int64)
    expected = np.array([[.15, 0], [0, .15]])
    np.testing.assert_allclose(strong._pose_cost(transformed, rows, strong.tracks), expected)
    np.testing.assert_array_equal(strong._pose_cost(transformed, rows, strong.tracks),
                                  deep._pose_cost(current, rows, deep.tracks).T)
    transformed.poses[:, 0, 2] = .299
    assert not strong._pose_cost(transformed, rows, strong.tracks).any()


def test_pose_uses_latest_real_observation_and_does_not_bypass_motion_gate() -> None:
    tracker = StrongSort(StrongSortConfig(pose_weight=.15))
    for index in range(3):
        tracker.update(two_people(index))
    change = two_people(3)
    change.poses[:, :, 0] += 1
    tracker.update(change)
    # Retaining an initial pose or taking an EMA would leave a nonzero diagonal.
    costs = tracker._pose_cost(change, np.array([0, 1], np.int64), tracker.tracks)
    np.testing.assert_array_equal(costs.diagonal(), [0, 0])
    far = two_people(4)
    far.boxes[:] += 1000
    far.poses[:, :, :2] += 1000
    assert not len(tracker.update(far).track_ids)


@pytest.mark.parametrize('weight', [-.1, 1.1, np.nan, np.inf])
def test_invalid_pose_weight_fails_explicitly(weight: float) -> None:
    with pytest.raises(ValueError, match='pose weight'):
        StrongSortConfig(pose_weight=weight)
