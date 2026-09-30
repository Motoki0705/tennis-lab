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
    assert unseen.selected_sides(document) == dict.fromkeys(unseen.CLIPS, [False, False, True])
    document['clips'][0]['annotation']['decided'] = False
    with pytest.raises(ValueError, match='Annotation-ball side'):
        unseen.selected_sides(document)
    document['clips'][0] = {'clip_id': unseen.CLIPS[0], 'observe_failed': {'status': 'failed'}}
    with pytest.raises(ValueError, match='court observation failure'):
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


def test_scoring_retains_abstained_clip_and_cannot_repeat(
    unseen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from src.tasks.player_association.evaluation.labels import (
        CameraLabels,
        ClipLabels,
        LabelledPerson,
    )
    from src.tennis_scene.pipeline.definition import file_identity
    from src.utils.geometry.triangulation import PinholeCamera

    scorer = importlib.import_module('person_unseen_score')
    cameras = [PinholeCamera(c, np.eye(3), np.eye(3), np.zeros(3)) for c in unseen.CAMERAS]
    calibration = SimpleNamespace(calibration=SimpleNamespace(views=[SimpleNamespace(camera=c) for c in cameras]))
    monkeypatch.setattr(scorer, 'ClipStore', lambda *a, **kw: SimpleNamespace(
        active=lambda _: object(), load=lambda *_: calibration))
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
        np.savez_compressed(root / 'predictions.npz', **arrays)
        (root / 'prediction.json').write_text(json.dumps({'status': 'ok' if index < 2 else 'undecided',
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
    assert result['tracking']['raw']['idf1'] == 1.
    assert result['tracking']['group']['idf1'] == 1.
    assert result['tracking']['associated']['idf1'] == .8
    assert result['tracking']['raw']['nonplayer_units_kept'] == 0
    assert (tmp_path / 'scoring/raw-camera-near-far.csv').is_file()
    with pytest.raises(FileExistsError, match='one batch'):
        scorer.score(tmp_path, label_path)
