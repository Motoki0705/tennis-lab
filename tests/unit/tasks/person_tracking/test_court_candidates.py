"""Dwell counts real ground observations; the cap never limits input persons."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.tasks.person_tracking.court_candidates import DwellConfig, select_candidates
from src.tasks.person_tracking.court_consistency import court_consistency
from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.selection_diagnosis import diagnose_tracks
from src.tasks.person_tracking.selection_metrics import aggregate_units, selection_units
from src.tasks.player_association.appearance.sampling import TrackAppearance
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


def test_post_association_3d_ground_check_uses_only_shared_observations() -> None:
    first = camera_tracks([(0., 5.)], np.ones((1, 2), bool))
    other = camera_tracks([(1., 5.)], np.ones((1, 2), bool))
    second = replace(other, camera=replace(other.camera, camera_id='cam1'))
    result = court_consistency([first, second], [np.array([[0, 0]]), np.array([[0, -1]])], FootpointConfig())
    assert len(result) == 1 and result[0]['shared_frames'] == 1
    assert result[0]['median_m'] == pytest.approx(1.)
    assert result[0]['p95_m'] == pytest.approx(1.)
    duplicate = camera_tracks([(0., 5.), (1., 5.)], np.ones((2, 2), bool))
    with pytest.raises(ValueError, match='same-camera identity overlap'):
        court_consistency([duplicate], [np.zeros((2, 2), np.int64)], FootpointConfig())


def test_fragments_accumulate_unique_dwell_before_six_chain_cap() -> None:
    seen: np.ndarray = np.zeros((8, 80), bool)
    for row in range(8):
        seen[row, row * 10:(row + 1) * 10] = True
    tracks = camera_tracks([(0., 5.)] * 8, seen)
    selected, diag = select_linked_candidates(tracks, 30., LinkingConfig(), FootpointConfig())
    assert selected.sum() == 80  # each fragment alone fails the 20-frame dwell
    assert diag['selected_groups'] == 1  # eight raw IDs are one candidate
    assert len(diag['links']) == 7
    assert diag['groups'][0]['in_core_frames'] == 80
    assert all(p['appearance'] == 'missing' for p in diag['links'])


def test_adjacent_only_track_cannot_qualify_or_borrow_dwell_across_jump() -> None:
    seen: np.ndarray = np.ones((3, 80), bool)
    tracks = camera_tracks([(0., 5.), (5.1, 5.), (6., 5.)], seen)
    boxes = tracks.boxes_xyxy.copy()
    boxes[0, 40:] = boxes[2, 40:]  # ID mixing to adjacent court
    tracks = replace(tracks, boxes_xyxy=boxes)
    selected, _ = select_linked_candidates(tracks, 30., LinkingConfig(), FootpointConfig())
    assert selected[0, :40].all() and not selected[0, 40:].any()
    assert not selected[1:].any()  # 5.1 is in outer corridor, outside dwell core


def test_appearance_vetoes_geometrically_close_fragments() -> None:
    seen: np.ndarray = np.zeros((2, 80), bool)
    seen[0, :15] = seen[1, 15:30] = True
    tracks = camera_tracks([(0., 5.)] * 2, seen)
    features = [TrackAppearance(np.array([frame], np.int64), np.array([vector], np.float32))
                for frame, vector in [(5, [1., 0.]), (20, [0., 1.])]]
    selected, diag = select_linked_candidates(replace(tracks, appearance=tuple(features)), 30., LinkingConfig(), FootpointConfig())
    assert not selected.any() and not diag['links']
    compatible = replace(tracks, appearance=(features[0], replace(features[1], embeddings=features[0].embeddings)))
    assert select_linked_candidates(compatible, 30., LinkingConfig(), FootpointConfig())[0].sum() == 30


def test_short_ambiguous_fragments_are_not_forced_and_handoff_counts_once() -> None:
    seen: np.ndarray = np.zeros((3, 80), bool)
    seen[0, :15] = seen[1:, 15:30] = True
    tracks = camera_tracks([(0., 5.), (.1, 5.), (-.1, 5.)], seen)
    selected, diag = select_linked_candidates(tracks, 30., LinkingConfig(), FootpointConfig())
    assert not selected.any()
    assert all(f['excluded_short_ambiguous'] for f in diag['fragments'])
    seen = np.zeros((2, 80), bool)
    seen[0, :15] = seen[1, 12:30] = True  # 0.1 s overlap
    selected, diag = select_linked_candidates(camera_tracks([(0., 5.)] * 2, seen), 30., LinkingConfig(), FootpointConfig())
    assert selected.sum() == 33 and selected.sum(0).max() == 2
    assert diag['groups'][0]['in_core_frames'] == 30
    # A longer overlap is not an allowed handoff.
    seen[1, 7:12] = True
    _, diag = select_linked_candidates(camera_tracks([(0., 5.)] * 2, seen), 30., LinkingConfig(), FootpointConfig())
    assert not diag['links']


def test_selected_player_keeps_wide_and_invalid_observations_without_lending_dwell() -> None:
    tracks = camera_tracks([(0., 5.), (6., 5.), (0., -30.), (5.1, 5.)], np.ones((4, 120), bool))
    boxes = tracks.boxes_xyxy.copy()
    # A continuous run from centre to beyond doubles width, then border loss.
    for frame in range(30, 90):
        fraction = (frame - 30) / 59
        boxes[0, frame] = (1 - fraction) * boxes[0, frame] + fraction * boxes[1, frame]
    boxes[0, 90:115] = boxes[1, 90:115]
    boxes[0, 115:] = boxes[2, 115:]
    selected, diag = select_linked_candidates(replace(tracks, boxes_xyxy=boxes), 30., LinkingConfig(), FootpointConfig())
    assert selected[0].all()  # region and projection validity never trim a player
    assert not selected[1:].any()  # no core dwell, including doubles-border person
    assert diag['selected_outside_corridor'] > 25
    assert diag['selected_invalid_footpoints'] == 5


def test_frame_noise_does_not_refragment_a_continuous_far_track() -> None:
    tracks = camera_tracks([(0., 5.), (0., 7.)], np.ones((2, 80), bool))
    boxes = tracks.boxes_xyxy[:1].copy()
    boxes[0, ::2] = tracks.boxes_xyxy[1, ::2]  # 2 m jitter in single-frame ground points
    noisy = replace(tracks, track_ids=tracks.track_ids[:1], boxes_xyxy=boxes, observed=tracks.observed[:1])
    selected, diag = select_linked_candidates(noisy, 30., LinkingConfig(), FootpointConfig())
    assert selected.sum() == 80
    assert len(diag['fragments']) == 1


def test_diagnosis_attributes_kept_duplicate_and_distinguishes_mixing_from_margin() -> None:
    tracks = camera_tracks([(6., 5.)] * 2, np.ones((2, 4), bool))
    labels = ClipLabels('video_000/clip_000', 4,
        (LabelledPerson('A', 'player', ''), LabelledPerson('X1', 'non_player', 'adjacent')),
        {'cam0': CameraLabels(np.arange(4, dtype=np.int64), np.array([0, 0, 1, 1], np.int64), tracks.boxes_xyxy[0])}, {})
    mask: np.ndarray = np.zeros((2, 4), bool)
    mask[1] = True  # rejected duplicate must not steal the selected unit
    diag = diagnose_tracks(tracks, mask, labels)
    assert diag['tracks'][0]['track_id'] == 1
    assert diag['tracks'][0]['composition'] == {'A': 2, 'X1': 2}
    assert diag['tracks'][0]['player_and_adjacent_mixed']
    assert diag['tracks'][0]['adjacent_inside_old'] == 2
    assert diag['tracks'][0]['adjacent_inside_corridor'] == 0


def test_subsecond_gap_links_but_does_not_fill_missing_observations() -> None:
    seen: np.ndarray = np.zeros((2, 240), bool)
    seen[0, :50] = seen[1, 75:125] = True
    tracks = camera_tracks([(0., 5.)] * 2, seen)
    selected, diag = select_linked_candidates(tracks, 30., LinkingConfig(), FootpointConfig())
    assert selected.sum() == 100 and not selected[:, 50:75].any()
    assert diag['selected_groups'] == 1
    assert not select_linked_candidates(tracks, 30., replace(LinkingConfig(), max_gap_s=.5), FootpointConfig())[0].any()


def test_ambiguous_predecessor_does_not_block_unique_successor_of_a_long_fragment() -> None:
    seen: np.ndarray = np.zeros((4, 150), bool)
    seen[:2, :40] = seen[2, 40:80] = seen[3, 80:100] = True
    tracks = camera_tracks([(-.1, 5.), (.1, 5.), (0., 5.), (0., 5.)], seen)
    selected, diag = select_linked_candidates(tracks, 30., LinkingConfig(), FootpointConfig())
    assert selected[3].sum() == 20
    assert any(set(g['track_ids']) == {2, 3} and g['selected'] for g in diag['groups'])


def test_association_timeline_joins_handoff_but_preserves_original_selection() -> None:
    from src.tasks.person_tracking.linked_timeline import linked_timeline
    seen: np.ndarray = np.zeros((2, 80), bool)
    seen[0, :15] = seen[1, 12:30] = True
    tracks = camera_tracks([(0., 5.)] * 2, seen)
    selected, diag = select_linked_candidates(tracks, 30., LinkingConfig(), FootpointConfig())
    timeline, origins = linked_timeline(tracks, diag)
    assert selected.sum() == 33 and timeline.observed.sum() == 30
    assert timeline.observed.shape == (1, 80)
    assert (origins[0, :15] == 0).all() and (origins[0, 15:30] == 1).all()
    assert (origins[0, 30:] == -1).all()


def test_wide_reference_denominator_comes_from_labels_even_when_untracked() -> None:
    tracks = camera_tracks([(6., 5.)], np.array([[True, False]]))
    labels = ClipLabels('video_000/clip_000', 2, (LabelledPerson('A', 'player', ''),),
        {'cam0': CameraLabels(np.array([0, 1], np.int64), np.array([0, 0], np.int64), tracks.boxes_xyxy[0])}, {})
    totals = aggregate_units(selection_units(tracks, tracks.observed, labels))
    assert totals['player_wide_units'] == 2
    assert totals['player_wide_tracked_03'] == totals['player_wide_kept_03'] == 1
