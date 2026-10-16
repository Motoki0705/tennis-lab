"""A partial-failure audit must reject corrupt or superseded evidence."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def collector(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('pipeline_stop_collection')


def detection() -> PersonDetectionOutput:
    return PersonDetectionOutput('cam0', np.array([0, 1], np.int64), np.array([[1, 2, 5, 8]], np.float32),
                                 np.array([.9], np.float32), np.array([0], np.int64))


def publish(store: ClipStore, node: str, version: int = 1, **dependencies: Any) -> Any:
    return store.publish(node, detection(), ArtifactCodec(PersonDetectionOutput), schema='person_detections',
                         version=2, identity={'revision': version}, dependencies=dependencies, provenance={})


def test_partial_read_checks_bytes_and_does_not_write(collector: Any, tmp_path: Path) -> None:
    store = ClipStore(tmp_path, {'clip_id': 'synthetic'})
    ref = publish(store, 'person_detection/cam0')
    before = {p: dual_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    _, outputs, _ = collector.load_completed(tmp_path)
    np.testing.assert_array_equal(outputs['person_detection/cam0'].boxes_xyxy, detection().boxes_xyxy)
    assert before == {p: dual_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    array = (tmp_path / ref.path).parent / 'array_0001.npy'
    np.save(array, np.zeros((1, 4), np.float32))
    with pytest.raises(ValueError, match='checksum mismatch'):
        collector.load_completed(tmp_path)


def test_partial_read_rejects_superseded_dependency(collector: Any, tmp_path: Path) -> None:
    store = ClipStore(tmp_path, {'clip_id': 'synthetic'})
    old = publish(store, 'person_detection/cam0')
    publish(store, 'person_detection/cam1', upstream=old)
    publish(store, 'person_detection/cam0', version=2)
    with pytest.raises(ValueError, match='superseded component dependency'):
        collector.load_completed(tmp_path)


@pytest.mark.parametrize('synthetic', [False, True])
def test_person_audit_rejects_invented_rows_or_synthetic_observation(collector: Any, synthetic: bool) -> None:
    track = SimpleNamespace(observed=np.ones((1, 1), bool),
        evidence=SimpleNamespace(detection_rows=np.array([[42]], np.int64)),
        reconstruction=SimpleNamespace(interpolated=np.full((1, 1), synthetic, bool)))
    outputs = {'person_detection/cam0': detection(), 'person_tracking/cam0': track,
               'player_selection/cam0': None, 'pose_estimation/cam0': None}
    with pytest.raises(ValueError, match='synthetic observation|non-source detection'):
        collector.audit_person('cam0', 1, outputs)
