from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import FramePoseCandidates, RowRejection
from src.tennis_scene.multiview_tracking.schema import (
    SampleEvent,
    TrackEnd,
    TrackingConfig,
)
from src.tennis_scene.multiview_tracking.tracking import track
from src.utils.geometry.triangulation import PointRejection


@dataclass(frozen=True)
class Person:
    x: float
    y: float
    cameras: tuple[bool, bool, bool] = (True, True, True)
    nose: bool = True
    ambiguous: bool = False


def candidates(timeline: list[list[Person]]) -> FramePoseCandidates:
    people = [p for frame in timeline for p in frame]
    count = len(people)
    offsets = np.cumsum([0, *(len(frame) for frame in timeline)], dtype=np.int64)
    positions: NDArray[np.float32] = np.zeros((count, 17, 3), np.float32)
    valid: NDArray[np.bool_] = np.zeros((count, 17), bool)
    support: NDArray[np.bool_] = np.asarray([p.cameras for p in people], bool).reshape(
        count, 3
    )
    for i, p in enumerate(people):
        valid[i, [5, 6, 11, 12]] = True
        positions[i, [5, 6, 11, 12]] = [
            [p.x - 0.2, p.y, 1.5],
            [p.x + 0.2, p.y, 1.5],
            [p.x - 0.15, p.y, 1.0],
            [p.x + 0.15, p.y, 1.0],
        ]
        if p.nose:
            valid[i, 0] = True
            positions[i, 0] = [p.x, p.y + 0.08, 1.7]
    ambiguous: NDArray[np.bool_] = np.asarray([p.ambiguous for p in people], bool)
    inliers = valid[:, None] & support[..., None]
    return FramePoseCandidates(
        ("cam2", "cam5", "cam0"),
        offsets,
        np.where(support, np.arange(count)[:, None], -1),
        positions,
        valid,
        np.where(valid, PointRejection.VALID, PointRejection.INSUFFICIENT_VIEWS).astype(
            np.uint8
        ),
        inliers,
        np.zeros(inliers.shape, np.float32),
        inliers,
        np.where(support.sum(axis=1) == 3, 2.9, 0.95).astype(np.float32),
        ambiguous,
        np.where(ambiguous, 0.01, 0).astype(np.float32),
        ambiguous,
        (offsets, offsets, offsets),
        tuple(
            np.where(
                rows, RowRejection.ASSIGNED, RowRejection.INSUFFICIENT_VIEWS
            ).astype(np.uint8)
            for rows in support.T
        ),
    )


def test_countermoving_people_keep_ids_across_windows():
    timeline = []
    for frame in range(30):
        first = Person(-1.05 + frame * 0.10, -3)
        second = Person(1.05 - frame * 0.10, -2.90)
        timeline.append([first, second] if frame % 2 == 0 else [second, first])
    result = track(candidates(timeline), fps=20, config=TrackingConfig(2))
    order = np.argsort(result.candidate_indices)
    expected = np.tile([0, 1, 1, 0], 15)
    np.testing.assert_array_equal(result.track_ids[order], expected)
    assert result.velocity_valid.all()
    speed = result.velocity_mps[order, 0]
    np.testing.assert_allclose(speed[expected == 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(speed[expected == 1], -2.0, atol=1e-5)


def test_identical_assignments_expose_temporal_tie_and_original_ambiguity():
    first, second = Person(0, -3, ambiguous=True), Person(0, -3, ambiguous=True)
    result = track(
        candidates([[first, second], [first, second]]), fps=10, config=TrackingConfig(2)
    )
    np.testing.assert_array_equal(result.track_ids, [0, 1, 0, 1])
    np.testing.assert_array_equal(
        result.event, [SampleEvent.BORN] * 2 + [SampleEvent.OBSERVED] * 2
    )
    np.testing.assert_array_equal(result.id_ambiguous, [False, False, True, True])
    np.testing.assert_allclose(result.temporal_margin, [-1, -1, 0, 0])
    np.testing.assert_allclose(result.frame_gap, [0.01] * 4)
    assert result.frame_ambiguous.all()


def test_a_support_change_keeps_one_id_without_fabricating_views():
    timeline = [
        [
            Person(
                frame * 0.02, 12 + (0 if frame < 10 else 0.26), (True, True, frame < 10)
            )
        ]
        for frame in range(20)
    ]
    inputs = candidates(timeline)
    result = track(inputs, fps=60, config=TrackingConfig(1))
    np.testing.assert_array_equal(result.track_ids, [0] * len(timeline))
    np.testing.assert_array_equal(result.source_rows[9:11], [[9, 9, 9], [10, 10, -1]])
    np.testing.assert_array_equal(result.positions_m, inputs.positions_m)
    assert result.selection.selected_track_ids == (0,)


def test_only_bracketed_short_gap_has_predicted_joints_and_same_id():
    timeline = [
        [Person(-0.3 + frame * 0.05, -3, nose=frame != 6)]
        if frame in (0, 1, 2, 3, 6, 12)
        else []
        for frame in range(13)
    ]
    inputs = candidates(timeline)
    result = track(inputs, fps=10, config=TrackingConfig(1))
    np.testing.assert_array_equal(
        np.flatnonzero(np.diff(result.frame_offsets)), [0, 1, 2, 3, 4, 5, 6, 12]
    )
    np.testing.assert_array_equal(result.track_ids, [0] * 7 + [1])
    np.testing.assert_array_equal(
        result.event,
        [
            SampleEvent.BORN,
            *[SampleEvent.OBSERVED] * 3,
            SampleEvent.OCCLUDED,
            SampleEvent.OCCLUDED,
            SampleEvent.REAPPEARED,
            SampleEvent.BORN,
        ],
    )
    np.testing.assert_array_equal(result.end_frames, [7, 13])
    np.testing.assert_array_equal(
        result.end_reasons, [TrackEnd.DISAPPEARED, TrackEnd.CLIP_BOUNDARY]
    )
    np.testing.assert_array_equal(
        result.positions_m[result.observed], inputs.positions_m
    )
    assert (result.source_rows[4:6] == -1).all()
    assert not result.measured_joints[4:6].any()
    assert result.predicted_joints[4:6].sum() == 8
    assert not result.predicted_joints[4:6, 0].any()
    np.testing.assert_allclose(result.positions_m[4:6, 5, 0], [-0.3, -0.25], atol=1e-6)
    assert result.selection.decisions[0].court_frames == 5


def test_selection_combines_residence_and_motion_with_doubles_width():
    timeline = []
    for frame in range(20):
        people = [Person(0.2, -5), Person(0.0, 6)]
        if frame < 5:
            people.extend([Person(5.485, -16.885), Person(5.49, 12)])
        if frame in (3, 4, 7, 8):
            people.append(Person(4.9, 15))
        people.append(Person(-4.7 + 0.06 * frame, 9))
        timeline.append(people)
    inputs = candidates(timeline)
    doubles = track(inputs, fps=10, config=TrackingConfig(players_per_side=2))
    singles = track(inputs, fps=10, config=TrackingConfig(players_per_side=1))
    # At equal residence, the moving trajectory outranks the stationary one.
    assert singles.selection.selected_track_ids == (0, 4)
    assert doubles.selection.selected_track_ids == (0, 1, 2, 4)
    assert doubles.selection.missing_per_side == (0, 0)
    evidence = doubles.selection.decisions
    assert evidence[2].court_frames == evidence[2].alley_frames == 5
    assert evidence[3].court_frames == 0
    assert (
        evidence[5].court_frames == 4 and evidence[5].reason == "insufficient_residence"
    )
    assert (
        (doubles.track_ids == 5) & (doubles.event == SampleEvent.OCCLUDED)
    ).sum() == 2


def test_empty_clip_reports_shortfall_and_invalid_mode_fails():
    result = track(candidates([[], [], []]), fps=29.97, config=TrackingConfig(2))
    np.testing.assert_array_equal(result.frame_offsets, [0, 0, 0, 0])
    assert result.selection.missing_per_side == (2, 2)
    with pytest.raises(ValueError, match="tracking settings"):
        TrackingConfig(3)
