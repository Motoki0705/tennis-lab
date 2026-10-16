from typing import Any

import numpy as np
import pytest

from src.tasks.person_tracking.contracts import DetectionFeatures, TrackAssignments
from src.tasks.person_tracking.evaluation import score_units, tracking_units
from src.tasks.person_tracking.feature_tracks import sampled_appearance, scatter_tracks
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.player_association.appearance.parts import NativeParts
from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.utils.geometry.triangulation import PinholeCamera


def test_duplicate_labels_are_one_unit_and_misses_remain_in_denominator() -> None:
    box = [10., 10., 30., 50.]
    labels = ClipLabels('clip', 2, (LabelledPerson('A', 'player', ''), LabelledPerson('X', 'non_player', '')),
        {'cam0': CameraLabels(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]),
                              np.array([box, box, [100, 0, 120, 40], box]))}, {})
    camera = PinholeCamera('cam0', np.eye(3), np.eye(3), np.zeros(3))
    tracks = CameraTracks(camera, (1920, 1080), np.array([1, 2, 3]),
        np.array([[box, box], [[100, 0, 120, 40]] * 2, [[500, 0, 520, 40]] * 2]), np.ones((3, 2), bool))
    selected = np.array([[True, False], [True, False], [True, False]])
    units, unknown = tracking_units(tracks, selected, labels)
    result = score_units(units)
    assert unknown == 1  # novel box is neither silently ignored nor claimed false
    assert result['player_units'] == 2
    assert (result['idtp'], result['idfp'], result['idfn']) == (1, 1, 1)
    assert result['idf1'] == .5


def test_global_identity_assignment_and_switch_fragment_are_different() -> None:
    units: list[dict[str, Any]] = [{'clip': 'c', 'camera': 'cam0', 'person': 'A', 'role': 'player',
        'near_far': 'near', 'frame': f, 'track_id': i} for f, i in enumerate([1, 1, None, 2, 2])]
    result = score_units(units)
    assert (result['idtp'], result['idfp'], result['idfn']) == (2, 2, 3)
    assert result['idf1'] == pytest.approx(4 / 9)
    assert result['id_switches'] == result['fragments'] == 1
    # A missing reference interval is not evidence of a tracking interruption.
    units[3]['frame'], units[4]['frame'] = 10, 11
    assert score_units(units)['fragments'] == 0


def test_camera_local_ids_are_not_shared_in_pooled_identity_assignment() -> None:
    units = [{'clip': 'c', 'camera': cam, 'person': person, 'role': 'player',
              'near_far': 'unknown', 'frame': 0, 'track_id': 1}
             for cam, person in [('cam0', 'A'), ('cam1', 'B')]]
    assert score_units(units)['idf1'] == 1.
    assert score_units([])['idf1'] is None


def test_feature_scatter_preserves_rows_and_rejects_neighbouring_box_substitution() -> None:
    camera = PinholeCamera('cam0', np.eye(3), np.eye(3), np.zeros(3))
    frame = DetectionFeatures(0, np.array([7, 9], np.int64),
        np.array([[10, 10, 30, 100], [100, 10, 130, 100]], np.float32), np.array([.9, .9], np.float32),
        np.zeros((2, 17, 3), np.float32), np.eye(2, dtype=np.float32), np.ones(2, bool))
    tracks, rows = scatter_tracks(camera, [frame], [TrackAssignments(0, np.array([9, 7]), np.array([1, 2]))], (1920, 1080))
    assert rows[:, 0].tolist() == [9, 7]
    appearances = sampled_appearance(tracks, rows, [frame])
    np.testing.assert_array_equal(appearances[0].embeddings, [[0, 1]])
    tracks.boxes_xyxy[0, 0, 0] += .01
    with pytest.raises(ValueError, match='exact source'):
        sampled_appearance(tracks, rows, [frame])


def test_group_idf1_uses_downstream_timeline_without_replacing_raw_metric() -> None:
    box = [10., 10., 30., 70.]
    camera = PinholeCamera('cam0', np.eye(3), np.eye(3), np.zeros(3))
    labels = ClipLabels('clip', 3, (LabelledPerson('A', 'player', ''),),
                       {'cam0': CameraLabels(np.array([0, 1, 2]), np.zeros(3, np.int64), np.array([box] * 3))}, {})
    observed = np.array([[True, True, False], [False, True, True]])
    appearances = tuple(TrackAppearance(np.flatnonzero(seen), np.empty((2, 0), np.float32),
                                       NativeParts(np.tile(np.array([[[1., 0.]]], np.float32), (2, 1, 1)), np.ones((2, 1), bool)))
                        for seen in observed)
    raw = CameraTracks(camera, (1920, 1080), np.array([1, 2]), np.array([[box] * 3] * 2), observed, appearances)
    selection = {'groups': [{'selected': True, 'fragments': [0, 1]}],
                 'fragments': [{'row': 0, 'track_id': 1, 'start': 0, 'end': 2}, {'row': 1, 'track_id': 2, 'start': 1, 'end': 3}]}
    group, origins = linked_timeline(raw, selection)
    assert origins.tolist() == [[0, 0, 1]]  # unchanged earlier-ID handoff rule
    assert group.appearance is not None and group.appearance[0].parts is not None
    assert group.appearance[0].frames.tolist() == [0, 1, 2]
    assert score_units(tracking_units(raw, observed, labels)[0])['idf1'] == pytest.approx(2 / 3)
    assert score_units(tracking_units(group, group.observed, labels)[0])['idf1'] == 1
