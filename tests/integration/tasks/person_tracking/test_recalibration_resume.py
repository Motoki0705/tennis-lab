"""Resume full cameras and fail on changed identity, rows, hashes or timelines."""
from __future__ import annotations

import copy
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.tasks.person_tracking.archive import save_features
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import FeatureConfig
from src.tennis_scene.pipeline.definition import file_identity
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def resume(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('association_recalibration_resume')


def fixture(tmp: Path) -> tuple[dict[str, Any], DetectionFeatures]:
    frame = DetectionFeatures(0, np.array([7], np.int64), np.array([[1, 2, 30, 60]], np.float32),
                              np.array([.9], np.float32), np.zeros((1, 17, 3), np.float32),
                              np.array([[1., 0.]], np.float32), np.array([True]))
    source = {'path': str(tmp / 'source.mp4'), 'sha256': 'video', 'camera_id': 'cam0',
              'num_frames': 1, 'fps': 25., 'width': 1920, 'height': 1080}
    plan = {'inputs': [{'key': 'complete', 'video': source}, {'key': 'partial', 'video': source}],
            'models': {}, 'production_tracking': {'encoder': 'synthetic'}, 'feature_config': asdict(FeatureConfig())}
    old_plan = tmp / 'old-plan.json'
    old_plan.write_text(json.dumps(plan))
    identity = file_identity(old_plan)
    det = tmp / 'detection.npz'
    np.savez(det, offsets=np.array([0, 1]), boxes=frame.boxes, scores=frame.scores, source_rows=frame.rows)
    feature = tmp / 'features.npz'
    save_features(feature, [frame], {'source': source, 'detection': file_identity(det), 'models': {},
                  'config': plan['feature_config'], 'encoder': 'synthetic', 'plan_sha256': identity['sha256']})
    progress = {'schema': 'i964_recalibration_features_v1', 'plan_sha256': identity['sha256'],
                'records': {'complete': {**file_identity(feature), 'frames': 1, 'detections': 1, 'bytes': feature.stat().st_size}},
                'detections': {'complete': file_identity(det), 'partial': file_identity(det)}}
    progress_path = tmp / 'old-progress.json'
    progress_path.write_text(json.dumps(progress))
    plan['resume'] = {'source_plan': identity, 'source_progress': file_identity(progress_path), 'reuse_keys': ['complete']}
    # An unlisted, plausible-looking partial file must not be discovered/reused.
    (tmp / 'partial.features.npz').write_bytes(feature.read_bytes())
    return plan, frame


def test_only_committed_whole_cameras_are_reused(resume: ModuleType, tmp_path: Path) -> None:
    plan, _ = fixture(tmp_path)
    records, detections = resume.resume_records(plan)
    assert list(records) == list(detections) == ['complete']
    # New final manifests keep the old per-camera provenance; no identity rewriting.
    final = {'schema': 'i964_recalibration_features_v1', 'plan_sha256': 'new',
             'records': records, 'detections': detections}
    assert resume.verify_completed(plan, final, 'new')[0] == records


@pytest.mark.parametrize('change', ['hash', 'identity', 'list', 'timeline', 'rows', 'provenance'])
def test_resume_stops_instead_of_recomputing_a_mismatch(resume: ModuleType, tmp_path: Path, change: str) -> None:
    plan, _ = fixture(tmp_path)
    old_path = Path(plan['resume']['source_progress']['path'])
    progress = json.loads(old_path.read_text())
    if change == 'hash':
        Path(progress['records']['complete']['path']).write_bytes(b'corrupt')
    elif change == 'identity':
        plan['feature_config']['min_appearance_height_px'] = 1.
    elif change == 'list':
        plan['resume']['reuse_keys'].append('partial')
    else:
        if change == 'timeline':
            progress['records']['complete']['frames'] = 2
        elif change == 'rows':
            progress['records']['complete']['detections'] = 2
        else:
            # Rehashing a modified source plan cannot excuse incompatible archive provenance.
            source_path = Path(plan['resume']['source_plan']['path'])
            source = json.loads(source_path.read_text())
            source['models'] = {'clip': {'sha256': 'changed'}}
            source_path.write_text(json.dumps(source))
            plan['models'] = source['models']
            plan['resume']['source_plan'] = file_identity(source_path)
            progress['plan_sha256'] = plan['resume']['source_plan']['sha256']
        old_path.write_text(json.dumps(progress))
        plan['resume']['source_progress'] = file_identity(old_path)
    with pytest.raises(ValueError):
        resume.resume_records(plan)


def test_extraction_skips_reused_camera_and_recomputes_partial(resume: ModuleType, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the actual producer with tiny fake models, without CUDA/video."""
    module = importlib.import_module('association_recalibration_features')
    plan, frame = fixture(tmp_path)
    plan.update(selected_clips=['video_000/clip_003'], dataset=str(tmp_path),
                pose_runtime={}, detector_runtime={}, merge_duplicates=False, budget={
                    'allocator_bytes': 1, 'disk_limit_bytes': 10_000_000})
    # Match identity additions in the parent (the synthetic data has no production assets).
    parent = copy.deepcopy(plan)
    parent.pop('resume')
    # Keep source plan/archives intact; bypass only the unrelated full real-data plan checks.
    verified = resume.resume_records({k: v for k, v in plan.items() if k not in (
        'selected_clips', 'dataset', 'pose_runtime', 'detector_runtime', 'merge_duplicates', 'budget')})
    monkeypatch.setattr(resume, 'resume_records', lambda _: copy.deepcopy(verified))
    report = tmp_path / 'new'
    report.mkdir()
    (report / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(module, 'validate_plan', lambda *_: None)
    monkeypatch.setattr(module, 'file_identity', lambda _: {})
    plan['models'] = dict.fromkeys(('detector', 'pose', 'clip', 'aflink'), {})
    (report / 'plan.json').write_text(json.dumps(plan))
    cfg = SimpleNamespace(people=SimpleNamespace(detector_checkpoint=None, vitpose_checkpoint=None,
        runtime=SimpleNamespace(vitpose=SimpleNamespace(flip_test=False, head=None), dino_detector={})),
        tracking_encoder_weights=None, aflink_checkpoint=None, merge_duplicate_person_boxes=False,
        tracking=SimpleNamespace(encoder='synthetic', identity=lambda: plan['production_tracking']))
    monkeypatch.setattr(module, 'runtime', lambda *_: cfg)
    monkeypatch.setattr(module, 'json_value', lambda _: {})
    for name in ('set_num_threads', 'set_num_interop_threads'):
        monkeypatch.setattr(module.torch, name, lambda *_: None)
    monkeypatch.setattr(module.torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(total_memory=10))
    for name in ('set_per_process_memory_fraction', 'empty_cache'):
        monkeypatch.setattr(module.torch.cuda, name, lambda *_: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(module.torch.cuda, name, lambda: 0)
    calls = []

    class Detector:
        def __init__(self, *_: Any, **__: Any) -> None:
            calls.append('detector')

        def process(self, _: Any) -> SimpleNamespace:
            return SimpleNamespace(frame_offsets=np.array([0, 1]), boxes_xyxy=frame.boxes,
                                   confidence=frame.scores, source_rows=frame.rows)

    monkeypatch.setattr(module, 'PersonDetectionModule', Detector)
    monkeypatch.setattr(module, 'ViTPosePose2D', lambda *a, **k: SimpleNamespace(unload=lambda: None))
    monkeypatch.setattr(module, 'build_encoder', lambda *a, **k: None)
    monkeypatch.setattr(module, 'UnpromptedEncoder', lambda *a: None)
    monkeypatch.setattr(module, 'FeatureExtractor', lambda *a: SimpleNamespace(extract=lambda *a: frame))
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', lambda *a, **k: [SimpleNamespace(index=0, frame=None)])
    module.extract(tmp_path, report)
    final = json.loads((report / 'features.json').read_text())
    assert calls == ['detector']
    assert final['status'] == 'ok' and set(final['records']) == {'complete', 'partial'}
    assert final['records']['complete'] == verified[0]['complete']
    assert final['reused_keys'] == ['complete'] and set(final['timings']) == {'partial'}
