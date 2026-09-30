"""One-shot gates, annotation-side provenance and a full tiny three-camera movie."""
import importlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def unseen(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('person_unseen')


def test_attempt_claim_is_immutable_even_after_failure(unseen: Any, tmp_path: Path) -> None:
    path = tmp_path / 'attempt.json'
    unseen.claim(path, {'inference_attempt': 1})
    with pytest.raises(FileExistsError):
        unseen.claim(path, {'inference_attempt': 2})
    assert json.loads(path.read_text()) == {'inference_attempt': 1}


def test_unpushed_freeze_prevents_opening_inputs(unseen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: Any) -> None:
        raise ValueError('not pushed')
    monkeypatch.setattr(unseen, 'require_pushed', reject)
    report = tmp_path / 'report'
    with pytest.raises(ValueError, match='not pushed'):
        unseen.plan(tmp_path / 'missing-freeze.json', 'not-pushed', report)
    assert not report.exists()


def test_only_decided_annotation_sides_are_accepted(unseen: Any) -> None:
    document = {'clips': [{'clip_id': c, 'camera_ids': list(unseen.CAMERAS),
                          'annotation': {'decided': True, 'view_half_turns': [False, False, True]},
                          'detector': {'decided': True, 'view_half_turns': [False, True, False]}}
                         for c in unseen.CLIPS]}
    assert unseen.selected_sides(document) == dict.fromkeys(unseen.CLIPS, {
        'status': 'ok', 'view_half_turns': [False, False, True]})
    document['clips'][0]['annotation']['decided'] = False
    with pytest.raises(ValueError, match='Annotation-ball side'):
        unseen.selected_sides(document)
    document['clips'][0] = {'clip_id': unseen.CLIPS[0], 'observe_failed': {'status': 'failed'}}
    stopped = unseen.selected_sides(document)[unseen.CLIPS[0]]
    assert stopped == {'status': 'stopped', 'reason': 'annotation_side_missing_after_court_failure',
                       'view_half_turns': None, 'observe_failed': {'status': 'failed'}}
    document['clips'].append(document['clips'][0])
    with pytest.raises(ValueError, match='Missing/duplicate'):
        unseen.selected_sides(document)


def test_video_contains_all_frames_of_three_cameras_and_no_labels(unseen: Any, tmp_path: Path) -> None:
    frames = 4
    videos = []
    arrays = {}
    for camera in unseen.CAMERAS:
        path = tmp_path / f'{camera}.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*'MJPG'), 12., (64, 36))
        assert writer.isOpened()
        for frame in range(frames):
            writer.write(np.full((36, 64, 3), frame * 30, np.uint8))
        writer.release()
        videos.append(SourceVideo(camera, path, dual_sha256(path), frames, 12., 64, 36))
        arrays.update({
            f'{camera}_boxes': np.tile(np.array([8, 4, 24, 30], np.float32), (1, frames, 1)),
            f'{camera}_observed': np.ones((1, frames), bool),
            f'{camera}_selected': np.ones((1, frames), bool),
            f'{camera}_ids': np.array([[0, 0, -1, 1]], np.int64),
            f'{camera}_track_ids': np.array([5], np.int64),
        })
    source = ClipSource('synthetic', tuple(videos))
    result = unseen.render(source, arrays, tmp_path / 'three.mp4', 'ok')
    assert result['frames_written'] == result['frames_read'] == frames
    assert result['labels_used'] is False
    assert result['cameras'] == list(unseen.CAMERAS)
    with pytest.raises(FileExistsError):
        unseen.render(source, arrays, tmp_path / 'three.mp4', 'ok')


@pytest.mark.parametrize('missing_camera', [None, 'cam1'])
def test_missing_side_preserves_selection_and_raw_tracks_without_associating(
    unseen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_camera: str | None,
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from src.tasks.player_association.association.config import load_association_config
    from tests.unit.tasks.person_tracking.test_court_candidates import camera_tracks

    tracks = camera_tracks([(0., 5.), (0., -5.), (9., 5.)], np.ones((3, 80), bool))
    source = ClipSource('synthetic', tuple(SourceVideo(c, tmp_path / f'{c}.mp4', 'unused', 80, 30., 1920, 1080)
                                        for c in unseen.CAMERAS))
    raw = SimpleNamespace(boxes_xyxy=tracks.boxes_xyxy, observed=tracks.observed,
                          track_ids=tracks.track_ids, evidence=object())
    runner = SimpleNamespace(output=lambda _: raw)
    cfg = SimpleNamespace(player_association=load_association_config(players_per_side=1), max_tracks_per_camera=6)
    cameras = {c: replace(tracks.camera, camera_id=c) for c in unseen.CAMERAS if c != missing_camera}
    monkeypatch.setattr(unseen, 'evidence_appearance', lambda *a: tracks.appearance)

    def forbidden(*args: Any) -> None:
        pytest.fail('Missing side or calibration must never call association')

    monkeypatch.setattr(unseen, 'associate', forbidden)
    side = {'status': 'stopped', 'reason': 'annotation_side_missing_after_court_failure', 'view_half_turns': None}
    result, arrays = unseen.predict(runner, source, cfg, side, cameras, tmp_path / 'predictions.npz')
    assert result['status'] == 'stopped' and result['view_half_turns'] is None
    for c in unseen.CAMERAS:
        np.testing.assert_array_equal(arrays[f'{c}_boxes'], tracks.boxes_xyxy)
        np.testing.assert_array_equal(arrays[f'{c}_observed'], tracks.observed)
        assert (arrays[f'{c}_ids'] == -1).all()
        assert (arrays[f'{c}_group_ids'] == -1).all()
        if c == missing_camera:
            assert not arrays[f'{c}_selected'].any()
            assert arrays[f'{c}_group_observed'].shape == (0, 80)
            assert result['selection'][c]['reason'] == 'camera_not_calibrated'
        else:
            assert arrays[f'{c}_selected'][:2].all() and not arrays[f'{c}_selected'][2].any()
            assert result['selection'][c]['orientation'] == 'camera_local_without_side'
    # The available-side path must use the exact same selection rule, including
    # half-turn invariance; incomplete calibration still cannot call associate.
    if missing_camera is None:
        monkeypatch.setattr(unseen, 'associate', lambda groups, *a: SimpleNamespace(
            player_ids=[np.zeros(g.observed.shape, np.int64) for g in groups], diagnostics={}))
    decided = {'status': 'ok', 'view_half_turns': [False, False, True]}
    second, oriented = unseen.predict(runner, source, cfg, decided, cameras, tmp_path / 'oriented.npz')
    assert second['status'] == ('ok' if missing_camera is None else 'stopped')
    for c in unseen.CAMERAS:
        for field in ('selected', 'group_boxes', 'group_observed', 'group_origins'):
            np.testing.assert_array_equal(arrays[f'{c}_{field}'], oriented[f'{c}_{field}'])


@pytest.mark.parametrize('expected_stop', [True, False])
def test_all_person_cameras_run_before_independent_court_failures(
    unseen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected_stop: bool,
) -> None:
    from types import SimpleNamespace

    calls: list[str] = []
    person_names = [f'{stage}/{c}' for c in unseen.CAMERAS for stage in ('person_detection', 'person_tracking')]
    nodes = [SimpleNamespace(name=name) for name in person_names + [f'court_detection/{c}' for c in unseen.CAMERAS]]

    class Runner:
        def __init__(self, selected: list[Any], store: Any) -> None:
            self.nodes = selected
            self.statuses: dict[str, str] = {}
            self.seconds: dict[str, float] = {}
            self.references: dict[str, Any] = {}
            self.active_node = None

        def run(self) -> None:
            for node in self.nodes:
                calls.append(node.name)
                if node.name == 'court_detection/cam1':
                    self.statuses[node.name] = 'failed'
                    message = 'No Court region meets model support/geometry requirements: []' if expected_stop else 'bad checkpoint'
                    raise ValueError(message)
                self.statuses[node.name] = 'executed'

        def output(self, name: str) -> str:
            return name.split('/')[1]

    def calibrate(camera: str, ids: tuple[str, ...], **kwargs: Any) -> Any:
        assert ids == (camera,)
        return SimpleNamespace(views=[], excluded={camera: 'no_accepted_homography'})

    monkeypatch.setattr(unseen, 'ComponentRunner', Runner)
    monkeypatch.setattr(unseen, 'audit_tracks', lambda *a: {'all_person_cameras_saved': True})
    monkeypatch.setattr(unseen, 'calibrate_local_courts', calibrate)
    # A plain mapping suffices for the synthetic calibration receipt.
    original_json = unseen.json_value
    monkeypatch.setattr(unseen, 'json_value', lambda value: vars(value) if isinstance(value, SimpleNamespace) else original_json(value))
    source, cfg = SimpleNamespace(size=(1920, 1080)), SimpleNamespace(camera_geometry=object())
    if expected_stop:
        _, cameras, audit = unseen.run_person_and_court(nodes, None, source, cfg, tmp_path)
        assert cameras == {} and audit == {'all_person_cameras_saved': True}
        assert calls == person_names + [f'court_detection/{c}' for c in unseen.CAMERAS]
        receipt = json.loads((tmp_path / 'court-execute.json').read_text())
        assert set(receipt) == set(unseen.CAMERAS)
        assert receipt['cam1']['reason'] == 'court_detection_unavailable'
    else:
        with pytest.raises(ValueError, match='bad checkpoint'):
            unseen.run_person_and_court(nodes, None, source, cfg, tmp_path)
    assert calls[:6] == person_names
    assert json.loads((tmp_path / 'person-execute.json').read_text())['statuses'] == dict.fromkeys(person_names, 'executed')


@pytest.mark.parametrize(('stop_status', 'uncalibrated'), [('undecided', False), ('stopped', False), ('stopped', True)])
def test_scoring_retains_abstained_clip_and_cannot_repeat(
    unseen: Any, tmp_path: Path, stop_status: str, uncalibrated: bool,
) -> None:
    from src.tasks.player_association.evaluation.labels import (
        CameraLabels,
        ClipLabels,
        LabelledPerson,
    )
    from src.tennis_scene.pipeline.definition import file_identity

    scorer = importlib.import_module('person_unseen_score')
    clips, records, labels_manifest = {}, [], {}
    for index, clip in enumerate(unseen.CLIPS):
        root = tmp_path / clip
        root.mkdir(parents=True)
        boxes = np.array([[8, 4, 20, 15], [25, 10, 40, 32], [50, 20, 60, 33]], np.float32)
        raw_boxes = np.repeat(boxes[:, None], 4, axis=1)
        label_cameras = {c: CameraLabels(np.repeat(np.arange(4, dtype=np.int64), 3),
                            np.tile(np.arange(3, dtype=np.int64), 4), np.tile(boxes.astype(float), (4, 1)))
                         for c in unseen.CAMERAS}
        labels = ClipLabels(clip, 4, (LabelledPerson('a', 'player', 'near'), LabelledPerson('b', 'player', 'far'),
                                     LabelledPerson('c', 'non_player', 'outside')), label_cameras, {'synthetic': True})
        labels.save(root / 'labels.json')
        labels_manifest[clip] = str(root / 'labels.json')
        arrays: dict[str, Any] = {}
        assigned = np.repeat(np.arange(2, dtype=np.int64)[:, None], 4, axis=1) if index < 2 else np.full((2, 4), -1, np.int64)
        for c in unseen.CAMERAS:
            arrays.update({
                f'{c}_boxes': raw_boxes, f'{c}_observed': np.ones((3, 4), bool),
                f'{c}_selected': np.repeat(np.array([[True], [True], [False]]), 4, axis=1),
                f'{c}_track_ids': np.arange(3, dtype=np.int64),
                f'{c}_group_boxes': raw_boxes[:2], f'{c}_group_observed': np.ones((2, 4), bool),
                f'{c}_group_track_ids': np.arange(2, dtype=np.int64),
                f'{c}_ids': np.concatenate((assigned, np.full((1, 4), -1, np.int64))),
                f'{c}_group_ids': assigned,
            })
            if index == 2 and uncalibrated:
                arrays[f'{c}_selected'][:] = False
                for field in ('group_boxes', 'group_observed', 'group_track_ids', 'group_ids'):
                    arrays[f'{c}_{field}'] = arrays[f'{c}_{field}'][:0]
        np.savez_compressed(root / 'predictions.npz', **arrays)
        (root / 'prediction.json').write_text(json.dumps({'status': 'ok' if index < 2 else stop_status,
            'reason': None if index < 2 else 'margin', 'arrays': file_identity(root / 'predictions.npz')}))
        clips[clip] = {'prediction': file_identity(root / 'prediction.json')}
        records.append({'clip': clip, 'source': {'videos': [{'num_frames': 4, 'width': 64, 'height': 36}]}})
    (tmp_path / 'complete.json').write_text(json.dumps({'status': 'ok', 'clips': clips}))
    (tmp_path / 'plan.json').write_text(json.dumps({'records': records}))
    label_path = tmp_path / 'labels-manifest.json'
    label_path.write_text(json.dumps(labels_manifest))
    scorer.score(tmp_path, label_path)
    result = json.loads((tmp_path / 'scoring/summary.json').read_text())
    assert result['association']['total_clips'] == 3
    assert result['association']['decided_clips'] == 2
    assert result['association']['pairs'] == {'tp': 48, 'fp': 0, 'fn': 24, 'f1': .8}
    assert result['tracking']['raw']['idf1'] == (.8 if uncalibrated else 1.)
    assert result['tracking']['group']['idf1'] == (.8 if uncalibrated else 1.)
    assert result['tracking']['associated']['idf1'] == .8
    assert result['tracking']['raw']['nonplayer_units_kept'] == 0
    assert (tmp_path / 'scoring/raw-camera-near-far.csv').is_file()
    with pytest.raises(FileExistsError, match='one batch'):
        scorer.score(tmp_path, label_path)
