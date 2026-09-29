import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tasks.person_tracking.archive import save_features
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def track(args: Namespace) -> None:
    spec = importlib.util.spec_from_file_location("feature_benchmark", PROJECT_ROOT / "tests/benchmarks/person_tracking_features.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.track(args)


def test_cpu_benchmark_replays_shared_features_and_rejects_changed_archive(tmp_path: Path) -> None:
    feature = DetectionFeatures(0, np.array([23], np.int64), np.array([[0, 0, 10, 20]], np.float32),
                  np.ones(1, np.float32), np.zeros((1, 17, 3), np.float32), np.array([[1, 0]], np.float32), np.ones(1, bool))
    path = tmp_path / 'cam0.features.npz'
    save_features(path, [feature], {'source': {'fps': 30}})
    manifest = {'status': 'ok', 'scope': 'prefix_smoke', 'cameras': {'cam0': {'path': str(path), 'sha256': dual_sha256(path)}}}
    (tmp_path / 'features.json').write_text(json.dumps(manifest))
    args = Namespace(report=tmp_path, method='botsort_pose')
    track(args)
    result = json.loads((tmp_path / 'tracking.botsort_pose.json').read_text())
    assert result['scope'] == 'prefix_smoke'
    assert result['cameras']['cam0']['assignments'][0]['detection_rows'] == [23]
    assert result['cameras']['cam0']['track_ids'] == [1]
    with pytest.raises(FileExistsError):
        track(args)
    (tmp_path / 'tracking.botsort_pose.json').unlink()
    # This test file is disposable; a mutated input must not be accepted under its old identity.
    with path.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        track(args)


def test_real_ultralytics_baseline_consumes_the_same_detection_archive(tmp_path: Path) -> None:
    video = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*'mp4v'), 30., (128, 128))
    assert writer.isOpened()
    writer.write(np.zeros((128, 128, 3), np.uint8))
    writer.release()
    feature = DetectionFeatures(0, np.array([23], np.int64), np.array([[10, 10, 30, 50]], np.float32),
        np.ones(1, np.float32), np.zeros((1, 17, 3), np.float32), np.array([[1, 0]], np.float32), np.ones(1, bool))
    path = tmp_path / 'cam0.features.npz'
    save_features(path, [feature], {'source': {'path': str(video), 'sha256': dual_sha256(video), 'fps': 30}})
    manifest = {'status': 'ok', 'scope': 'prefix_smoke', 'cameras': {'cam0': {'path': str(path), 'sha256': dual_sha256(path)}}}
    (tmp_path / 'features.json').write_text(json.dumps(manifest))
    for method in ('botsort_pose', 'ultralytics_botsort'):
        track(Namespace(report=tmp_path, method=method))
    baseline = json.loads((tmp_path / 'tracking.ultralytics_botsort.json').read_text())
    derivative = json.loads((tmp_path / 'tracking.botsort_pose.json').read_text())
    assert baseline['features_sha256'] == derivative['features_sha256']
    assert baseline['cameras']['cam0']['track_ids'] == [1]
    assert baseline['cameras']['cam0']['observed_detections'] == 1
    assert baseline['config']['gmc_method'] == 'sparseOptFlow' and baseline['config']['with_reid'] is False
