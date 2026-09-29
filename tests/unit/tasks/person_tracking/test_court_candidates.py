"""Dwell counts real ground observations; the cap never limits input persons."""
from __future__ import annotations

import numpy as np

from src.tasks.person_tracking.court_candidates import DwellConfig, select_candidates
from src.tasks.person_tracking.selection_metrics import aggregate_units, selection_units
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tasks.player_association.geometry.region import PlayRegionConfig
from src.utils.geometry.triangulation import PinholeCamera


def camera_tracks(positions: list[tuple[float, float]], observed: np.ndarray) -> CameraTracks:
    rotation = np.diag([1., -1., -1.])
    camera = PinholeCamera('cam0', np.array([[1000., 0, 960], [0, 1000., 540], [0, 0, 1]]),
                           rotation, -rotation @ np.array([0., 0., 30.]))
    bottom, _ = camera.project(np.column_stack((positions, np.zeros(len(positions)))))
    boxes = np.stack((bottom[:, 0] - 10, bottom[:, 1] - 60, bottom[:, 0] + 10, bottom[:, 1]), axis=-1)
    return CameraTracks(camera, (1920, 1080), np.arange(len(positions), dtype=np.int64),
                        np.repeat(boxes[:, None], observed.shape[1], axis=1), observed)


def test_dwell_ignores_span_and_caps_only_eligible_candidates() -> None:
    observed: np.ndarray = np.ones((10, 20), bool)
    observed[8] = False
    observed[8, [0, 19]] = True  # 20-frame span, only two observed frames
    tracks = camera_tracks([(float(x), 0.) for x in range(-4, 5)] + [(20., 0.)], observed)
    chosen, records = select_candidates(tracks, DwellConfig(PlayRegionConfig(2.5, 5), FootpointConfig(), .25))
    assert chosen.tolist() == list(range(6))
    assert len(records) == 10  # no input cap, no ID recycling
    assert records[6]['reason'] == 'candidate_cap'
    assert records[8]['in_region_frames'] == 2
    assert records[8]['reason'] == records[9]['reason'] == 'insufficient_dwell'


def test_court_footpoint_and_border_validity_control_selection() -> None:
    tracks = camera_tracks([(0., 5.), (9., 5.), (0., -30.)], np.ones((3, 10), bool))
    chosen, records = select_candidates(tracks, DwellConfig(PlayRegionConfig(2.5, 5), FootpointConfig(), .25))
    assert chosen.tolist() == [0]
    assert records[1]['in_region_frames'] == 0  # adjacent to court
    assert records[2]['valid_footpoint_frames'] == 0  # feet below image boundary


def test_missing_nonplayer_is_not_counted_as_geometric_rejection() -> None:
    tracks = camera_tracks([(0., 5.)], np.ones((1, 2), bool))
    box = tracks.boxes_xyxy[0, 0]
    old = CameraLabels(np.array([0, 0, 0, 1, 1], np.int64), np.array([0, 0, 1, 0, 1], np.int64),
        np.stack([box, box, box + [200, 0, 200, 0], box, box + [200, 0, 200, 0]]))
    labels = ClipLabels('video_000/clip_000', 2,
        (LabelledPerson('A', 'player', 'player'), LabelledPerson('X1', 'non_player', 'adjacent')),
        {'cam0': old}, {})
    units = selection_units(tracks, np.ones((1, 2), bool), labels)
    result = aggregate_units(units)
    assert result['player_units'] == result['player_kept_05'] == 2  # duplicate boxes count once
    assert result['non_player_excluded_05'] == 2
    assert result['non_player_rejected_among_tracked_05'] == 0
    assert result['player_identities_kept50_05'] == result['non_player_identities_rejected_all_05'] == 1
