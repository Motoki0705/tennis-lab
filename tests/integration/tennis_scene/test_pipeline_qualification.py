"""Fresh-store/load-process gates and full-frame video on tiny local inputs."""
from __future__ import annotations

import importlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def qualification(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('pipeline_qualification')


def empty_scene(frames: int = 3) -> SceneResult:
    return SceneResult(num_frames=frames, fps=12., width=64, height=36,
        court_kp=np.zeros((3, frames, 14, 2), np.float32), court_vis=np.zeros((3, frames, 14), np.float32),
        player_position=np.zeros((1, frames, 3), np.float32), player_yaw=np.zeros((1, frames), np.float32),
        ball_uv=np.zeros((3, frames, 2), np.float32), ball_vis=np.zeros((3, frames), bool),
        ball_3d=np.zeros((frames, 3), np.float32), human_kp_2d=np.full((1, 3, frames, 17, 2), .5, np.float32),
        human_kp_vis=np.ones((1, 3, frames, 17), np.float32), player_track_ids=np.array([0], np.int32),
        player_kp_3d=np.zeros((1, frames, 17, 3), np.float32), player_observed=np.zeros((1, frames), bool),
        player_valid=np.zeros((1, frames), bool), player_heading_valid=np.zeros((1, frames), bool),
        player_kp_3d_vis=np.zeros((1, frames, 17), bool), player_smpl_valid=np.zeros((1, frames), bool),
        ball_3d_valid=np.zeros(frames, bool), player_rejection_code=np.ones((1, frames), np.uint8),
        player_kp_3d_rejection_code=np.ones((1, frames, 17), np.uint8), ball_rejection_code=np.ones(frames, np.uint8),
        metadata={'scene_schema_version': 2})


def source_videos(root: Path, frames: int = 3) -> ClipSource:
    videos = []
    for camera in ('cam0', 'cam1', 'cam2'):
        path = root / f'{camera}.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*'MJPG'), 12., (64, 36))
        assert writer.isOpened()
        for frame in range(frames):
            writer.write(np.full((36, 64, 3), frame * 30, np.uint8))
        writer.release()
        videos.append(SourceVideo(camera, path, dual_sha256(path), frames, 12., 64, 36))
    return ClipSource('synthetic', tuple(videos))


def test_qualification_rejects_cached_or_partial_execution(qualification: Any) -> None:
    order = ['person_tracking/cam0', 'person_tracking/cam1', 'person_tracking/cam2']
    qualification.require_statuses(dict.fromkeys(order, 'executed'), order, 'executed')
    for actual in ({order[0]: 'executed'}, dict.fromkeys(order, 'cached'), {}):
        with pytest.raises(ValueError, match='All planned'):
            qualification.require_statuses(actual, order, 'executed')
    with pytest.raises(ValueError, match='loaded'):
        qualification.require_statuses(dict.fromkeys(order, 'executed'), order, 'loaded')


def test_fresh_store_and_process_required_before_loading_inputs(qualification: Any, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qualification, 'load_inputs', lambda *_: pytest.fail('must stop before inputs'))
    (tmp_path / 'store').mkdir()
    with pytest.raises(FileExistsError, match='fresh store'):
        qualification.execute(tmp_path)
    for execution in ({'status': 'failed', 'pid': -1}, {'status': 'ok', 'pid': os.getpid()}):
        (tmp_path / 'execute.json').write_text(json.dumps(execution))
        with pytest.raises(ValueError, match='different process'):
            qualification.validate(tmp_path)


def test_archive_all_fields_comparison_detects_changed_observations(qualification: Any, tmp_path: Path) -> None:
    expected = empty_scene()
    save_scene_result(expected, tmp_path / 'scene.npz')
    restored = load_scene_result(tmp_path / 'scene.npz')
    qualification.same_scene(expected, restored)
    assert restored.human_kp_2d is not None
    restored.human_kp_2d[0, 0, 0, 0, 0] = .75
    with pytest.raises(AssertionError, match='human_kp_2d'):
        qualification.same_scene(expected, restored)


def test_video_preserves_all_three_camera_timelines(qualification: Any, tmp_path: Path) -> None:
    source = source_videos(tmp_path)
    record = qualification.render(empty_scene(), source, tmp_path / 'full.mp4')
    assert record['frames_written'] == record['frames_read'] == 3
    assert record['fps'] == 12. and record['camera_ids'] == ['cam0', 'cam1', 'cam2']
    with pytest.raises(FileExistsError):
        qualification.render(empty_scene(), source, tmp_path / 'full.mp4')


def test_video_stops_on_truncated_camera(qualification: Any, tmp_path: Path) -> None:
    source = source_videos(tmp_path)
    declared = ClipSource(source.clip_id, tuple(replace(v, num_frames=4) for v in source.videos))
    with pytest.raises(ValueError, match='ended/changed'):
        qualification.render(empty_scene(4), declared, tmp_path / 'incomplete.mp4')
    with pytest.raises(ValueError, match='three complete'):
        qualification.render(empty_scene(), ClipSource('two', source.videos[:2]), tmp_path / 'two.mp4')
