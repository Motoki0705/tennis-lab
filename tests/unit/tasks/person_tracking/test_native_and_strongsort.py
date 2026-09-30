from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tasks.person_tracking.archive import save_features
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.deep_ocsort_pose import DeepOCSortPose
from src.tasks.person_tracking.strongsort import (
    InvalidPrediction,
    MotionState,
    StrongSort,
)
from src.tasks.person_tracking.strongsort_offline import (
    aflink_inputs,
    gaussian_interpolation,
)
from src.tasks.player_association.appearance.affinity import (
    AppearanceAffinityConfig,
    appearance_evidence,
    segment_appearance,
)
from src.tasks.player_association.appearance.parts import (
    NativeParts,
    mean_parts,
    part_distance,
    update_parts,
)
from src.tasks.player_association.appearance.sampling import TrackAppearance


def frame(index: int, *, empty: bool = False, native: bool = False) -> DetectionFeatures:
    n = 0 if empty else 1
    parts = NativeParts(np.tile(np.array([[[1, 0], [0, 1]]], np.float32), (n, 1, 1)), np.ones((n, 2), bool)) if native else None
    return DetectionFeatures(index, np.arange(index, index + n, dtype=np.int64),
                             np.tile(np.array([[10, 10, 30, 70]], np.float32), (n, 1)), np.full(n, .9, np.float32),
                             np.zeros((n, 17, 3), np.float32), np.zeros((n, 2), np.float32) if native else np.tile(np.array([[1, 0]], np.float32), (n, 1)),
                             np.full(n, not native, bool), parts)


def test_native_distance_is_mean_euclidean_on_intersection_and_has_missing_mask() -> None:
    a = NativeParts(np.array([[[1, 0], [1, 0]]], np.float32), np.array([[True, True]]))
    b = NativeParts(np.array([[[0, 1], [-1, 0]]], np.float32), np.array([[True, False]]))
    distance, valid = part_distance(a, b)
    assert valid.item() and distance.item() == pytest.approx(np.sqrt(2))
    # Changing an invisible part must not alter the distance.
    b.embeddings[0, 1] = [0, 1]
    assert part_distance(a, b)[0].item() == distance.item()
    missing = NativeParts(a.embeddings, np.array([[False, True]]))
    assert not part_distance(missing, b)[1].item()
    assert appearance_evidence(missing, b, AppearanceAffinityConfig('kpr', 62.7, .847, 10))['appearance_missing'] == 'no_common_visible_parts'


def test_native_ema_and_segment_average_ignore_invisible_samples() -> None:
    previous = NativeParts(np.array([[[1, 0], [0, 1]]], np.float32), np.array([[True, False]]))
    current = NativeParts(np.array([[[-1, 0], [0, -1]]], np.float32), np.array([[False, True]]))
    updated = update_parts(previous, current, .9)
    np.testing.assert_array_equal(updated.embeddings, [[[1, 0], [0, -1]]])
    assert updated.visible.all()
    samples = NativeParts(np.concatenate((previous.embeddings, current.embeddings)), np.concatenate((previous.visible, current.visible)))
    np.testing.assert_array_equal(mean_parts(samples).embeddings, updated.embeddings)
    sampled = TrackAppearance(np.array([3, 10]), np.zeros((2, 0), np.float32), samples)
    early = segment_appearance(sampled, 0, 5)
    assert isinstance(early, NativeParts)
    np.testing.assert_array_equal(early.visible, previous.visible)
    with pytest.raises(ValueError, match='whole-image'):
        appearance_evidence(updated, np.array([1., 0.]), AppearanceAffinityConfig('kpr', 62.7, .847, 10))


@pytest.mark.parametrize('native', [False, True])
def test_strongsort_confirmation_and_gap_recovery_preserve_real_rows(native: bool) -> None:
    tracker = StrongSort()
    assert not len(tracker.update(frame(0, native=native)).track_ids)
    assert not len(tracker.update(frame(1, native=native)).track_ids)
    assert tracker.update(frame(2, native=native)).track_ids.tolist() == [1]
    assert not len(tracker.update(frame(3, empty=True, native=native)).track_ids)
    assert not len(tracker.update(frame(4, empty=True, native=native)).track_ids)
    result = tracker.update(frame(5, native=native))
    assert result.track_ids.tolist() == [1]
    assert result.detection_rows.tolist() == [5]


def test_strongsort_motion_gate_rejects_same_appearance_at_impossible_position() -> None:
    tracker = StrongSort()
    for index in range(3):
        tracker.update(frame(index))
    shifted = frame(3)
    shifted.boxes[:] += 1000
    assert not len(tracker.update(shifted).track_ids)
    assert tracker.tracks[-1].identity == 2


def test_nsa_covariance_gives_high_confidence_measurement_more_weight() -> None:
    box = np.array([0, 0, 20, 60], np.float32)
    low, high = MotionState.initiate(box), MotionState.initiate(box)
    low.predict()
    high.predict()
    low.update(box + 5, .1)
    high.update(box + 5, .9)
    assert high.mean[0] > low.mean[0]
    assert np.linalg.eigvalsh(high.covariance).min() > 0


def test_invalid_prediction_fails_with_frame_and_id_without_silently_dropping_track() -> None:
    tracker = StrongSort()
    tracker.update(frame(0))
    tracker.tracks[0].motion.mean[6] = -10
    with pytest.raises(InvalidPrediction, match='frame=1 track=1'):
        tracker.update(frame(1))
    assert len(tracker.tracks) == 1


def test_deep_native_parts_survive_gap_and_cannot_be_silently_saved_as_v2(tmp_path: Path) -> None:
    tracker = DeepOCSortPose(30.)
    assert tracker.update(frame(0, native=True)).track_ids.tolist() == [1]
    assert tracker.update(frame(1, native=True)).track_ids.tolist() == [1]
    with pytest.raises(ValueError, match='representation changed'):
        tracker.update(frame(2))
    with pytest.raises(ValueError, match='cannot discard'):
        save_features(tmp_path / 'bad.npz', [frame(0, native=True)], {})
    with pytest.raises(ValueError, match='coexist'):
        replace(frame(0), parts=frame(0, native=True).parts)


def test_aflink_preprocessing_endpoint_padding_and_shared_scale() -> None:
    left = np.array([[2, 10, 100], [3, 20, 200]], float)
    right = np.array([[4, 30, 300], [5, 40, 400]], float)
    a, b = aflink_inputs(left, right)
    assert a.shape == b.shape == (1, 1, 30, 3)
    np.testing.assert_allclose(a.numpy()[0, 0, -2:], [[-.2, -.5, -.5], [.2, 0, 0]], atol=1e-5)
    np.testing.assert_allclose(b.numpy()[0, 0, :2], [[.6, .5, .5], [1, 1, 1]], atol=1e-5)
    assert float(a[0, 0, 0, 0]) < -.99


def test_gsi_never_promotes_synthetic_observations_and_does_not_bridge_long_gaps() -> None:
    boxes: np.ndarray = np.zeros((1, 30, 4), np.float32)
    mask: np.ndarray = np.zeros((1, 30), bool)
    for time in (0, 2, 5, 29):
        boxes[0, time] = [time, 10, time + 20, 70]
        mask[0, time] = True
    result = gaussian_interpolation(boxes, mask)
    np.testing.assert_array_equal(result.observed, mask)
    assert np.flatnonzero(result.interpolated[0]).tolist() == [1, 3, 4]
    assert not (result.interpolated & result.observed).any()
    np.testing.assert_allclose(result.boxes[0, 1], [1, 10, 21, 70], atol=.01)
