"""Fresh all-execute qualification, followed by a separate load-only process.

Consumes pipeline_preflight.py's pinned config. No label reader or imported
artifacts are used. A failed stage keeps its receipt and cannot be retried into
the same store.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pipeline_preflight import CODE, check_full_recipe  # type: ignore[import-not-found]
from pipeline_qualification_video import render  # type: ignore[import-not-found]

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import file_identity, standard_definition
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.schema import SceneResult, validate_scene_result_arrays
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole


def require_statuses(statuses: dict[str, str], order: list[str], expected: str) -> None:
    if set(statuses) != set(order) or not statuses or any(s != expected for s in statuses.values()):
        raise ValueError(f'All planned components must be {expected}: {statuses}')


def same_scene(left: SceneResult, right: SceneResult) -> None:
    """Compare every persisted field, including validity masks and metadata."""
    validate_scene_result_arrays(left)
    validate_scene_result_arrays(right)
    for field in fields(SceneResult):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.dtype != b.dtype:
                raise ValueError(f'Scene dtype changed: {field.name}')
            np.testing.assert_array_equal(a, b, err_msg=field.name)
        elif a != b:
            raise ValueError(f'Scene field changed: {field.name}')


def load_inputs(report: Path) -> tuple[PipelineRuntimeConfig, ClipSource, dict[str, Any]]:
    preflight = json.loads((report / 'preflight.json').read_text())
    plan = json.loads((report / 'plan.json').read_text())
    if preflight['schema'] != 'full_pipeline_preflight_v1' or preflight['status'] != 'ok':
        raise ValueError('A successful CPU preflight is required')
    for record in (plan['preflight'], preflight['config'], *preflight['assets'].values(),
                   *plan['entrypoints']):
        if file_identity(Path(record['path'])) != record:
            raise ValueError(f'Pinned input changed: {record["path"]}')
    current = {str(p.relative_to(CODE)): dual_sha256(p) for p in sorted((CODE / 'src').rglob('*.py'))}
    if current != preflight['code']:
        raise ValueError('Source code changed since qualification preflight')
    config = OmegaConf.load(preflight['config']['path'])
    if not isinstance(config, DictConfig):
        raise TypeError('Expected a pipeline config mapping')
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=True)
    check_full_recipe(runtime)
    source = build_clip_source(runtime.video_paths, runtime.camera_ids, clip_id=preflight['source']['clip_id'])
    if json_value(source) != preflight['source'] or source.camera_ids != ('cam0', 'cam1', 'cam2'):
        raise ValueError('The full three-camera timeline changed')
    return runtime, source, preflight


def execute(report: Path) -> None:
    if (report / 'store').exists() or (report / 'execute.json').exists():
        raise FileExistsError('Qualification execution needs a fresh store and receipt')
    runtime, source, preflight = load_inputs(report)
    budget = json.loads((report / 'plan.json').read_text())['budget']
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(budget['allocator_bytes'] / total, 0)
    application = TennisSceneOrchestrator(runtime)
    receipt: dict[str, Any] = {'status': 'running', 'pid': os.getpid(), 'preflight': file_identity(report / 'preflight.json'),
                               'imported_nodes': [], 'dev_labels_opened': False}
    try:
        scene = application.run(runtime.video_paths, video_role=PathRole.DATA, camera_ids=source.camera_ids,
                                store_root=report / 'store', clip_id=source.clip_id)
        assert application.last_runner is not None
        require_statuses(application.last_runner.statuses, preflight['order'], 'executed')
        validate_scene_result_arrays(scene)
        receipt.update(status='ok', scene_index=file_identity(report / 'store/scene.json'),
                       frames=scene.num_frames, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['run'] = application.last_receipt
        write_json_atomic(report / 'execute.json', receipt)


def audit_outputs(runner: ComponentRunner, source: ClipSource, scene: SceneResult) -> dict[str, Any]:
    """Check full timelines, actual source rows, GSI separation and selected axes."""
    identities = runner.output('player_association')
    if identities.camera_ids != source.camera_ids or scene.num_frames != source.num_frames:
        raise ValueError('Scene/identity camera or frame coverage changed')
    cameras = {}
    for view, camera in enumerate(source.camera_ids):
        detections = runner.output(f'person_detection/{camera}')
        tracks = runner.output(f'person_tracking/{camera}')
        selection = runner.output(f'player_selection/{camera}')
        ball = runner.output(f'ball_detection/{camera}')
        if len(detections.frame_offsets) != source.num_frames + 1 or detections.duplicate_merges \
                or detections.source_rows is None or tracks.observed.shape[1] != source.num_frames \
                or tracks.evidence is None or tracks.reconstruction is None:
            raise ValueError(f'{camera}: incomplete production detection/tracking')
        if (detections.confidence < np.float32(.3)).any() or (tracks.reconstruction.interpolated & tracks.observed).any():
            raise ValueError(f'{camera}: detector gate or synthetic observation changed')
        for frame in range(source.num_frames):
            start, end = detections.frame_offsets[frame:frame + 2]
            rows = tracks.evidence.detection_rows[:, frame][tracks.observed[:, frame]]
            source_rows = detections.source_rows[start:end]
            indices = np.searchsorted(source_rows, rows)
            if (indices >= len(source_rows)).any() or not np.array_equal(source_rows[indices], rows):
                raise ValueError(f'{camera}: tracking cites a non-source detection')
            np.testing.assert_array_equal(tracks.boxes_xyxy[:, frame][tracks.observed[:, frame]],
                                          detections.boxes_xyxy[start:end][indices])
        selected = selection.tracks
        if len(selected.track_ids) > 6 or selected.observed.shape[1] != source.num_frames:
            raise ValueError(f'{camera}: selection cap/timeline changed')
        np.testing.assert_array_equal(identities.local_track_ids[view, :len(selected.track_ids)], selected.track_ids)
        if ball.observed.shape != (source.num_frames,) or ball.evidence is None:
            raise ValueError(f'{camera}: missing full-frame model ball evidence')
        cameras[camera] = {'frames': source.num_frames, 'detections': len(detections.confidence),
                          'tracks': len(tracks.track_ids), 'real_observations': int(tracks.observed.sum()),
                          'gsi_synthetic': int(tracks.reconstruction.interpolated.sum()),
                          'selected_groups': len(selected.track_ids), 'ball_observed': int(ball.observed.sum())}
    expected = sorted(int(p) for p in np.unique(identities.player_ids) if p >= 0)
    if scene.player_track_ids is None or scene.player_track_ids.tolist() != expected or len(expected) != 2:
        raise ValueError('clip_000 must reconstruct exactly its two associated players')
    for node, field in (('body_view_selection', 'selections'), ('gvhmr', 'bodies')):
        if sorted(item.person_id for item in getattr(runner.output(node), field)) != expected:
            raise ValueError(f'{node}: body IDs differ from association')
    return {'cameras': cameras, 'player_ids': expected,
            'court_side': json_value(runner.output('court_side')),
            'association': identities.diagnostics, 'validity': scene.metadata['validity_statistics']}


def validate(report: Path) -> None:
    if (report / 'qualification.json').exists():
        raise FileExistsError('Qualification validation receipts are immutable')
    executed = json.loads((report / 'execute.json').read_text())
    if executed['status'] != 'ok' or executed['pid'] == os.getpid():
        raise ValueError('Validation requires successful execution in a different process')
    runtime, source, preflight = load_inputs(report)
    if file_identity(report / 'store/scene.json') != executed['scene_index']:
        raise ValueError('Store index changed after execution')
    application = TennisSceneOrchestrator(runtime)
    nodes = standard_definition(runtime, source, code_identity=application.code_identity)
    store = ClipStore(report / 'store', json_value(source), memory_entries=0)
    runner = ComponentRunner([replace(node, source='load') for node in nodes], store)
    receipt: dict[str, Any] = {'status': 'running', 'pid': os.getpid(), 'execute_pid': executed['pid'],
                               'execute': file_identity(report / 'execute.json'), 'dev_labels_opened': False}
    try:
        runner.run()
        require_statuses(runner.statuses, preflight['order'], 'loaded')
        scene = load_scene_result(store.index_path)
        same_scene(scene, runner.output('scene_assembly'))
        receipt['outputs'] = audit_outputs(runner, source, scene)
        receipt['video'] = render(scene, source, report / 'three_camera_full.mp4')
        receipt.update(status='ok', frames=source.num_frames, artifacts=json_value(runner.references))
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(load_only_statuses=runner.statuses, load_only_seconds=runner.seconds)
        write_json_atomic(report / 'qualification.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--phase', choices=('execute', 'validate'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    (execute if args.phase == 'execute' else validate)(args.report.resolve())
