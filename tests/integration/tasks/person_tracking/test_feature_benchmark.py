import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from src.tasks.person_tracking.archive import save_features
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.utils.checksum import dual_sha256
from tests.benchmarks.person_tracking_features import track


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
