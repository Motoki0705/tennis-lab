import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.person_tracking.archive import load_features, save_features
from src.tasks.person_tracking.contracts import DetectionFeatures


def test_archive_roundtrip_keeps_empty_frame_and_noncontiguous_row_identity(tmp_path: Path) -> None:
    observed = DetectionFeatures(0, np.array([7, 11], np.int64), np.array([[0, 0, 10, 20]] * 2, np.float32),
                                 np.ones(2, np.float32), np.zeros((2, 17, 3), np.float32),
                                 np.array([[1, 0], [0, 1]], np.float32), np.ones(2, bool))
    empty = DetectionFeatures(1, np.empty(0, np.int64), np.empty((0, 4), np.float32), np.empty(0, np.float32),
                              np.empty((0, 17, 3), np.float32), np.empty((0, 2), np.float32), np.empty(0, bool))
    path = tmp_path / 'features.npz'
    save_features(path, [observed, empty], {'detection_sha256': 'a' * 64})
    frames, provenance = load_features(path)
    assert frames[0].rows.tolist() == [7, 11] and frames[1].embeddings.shape == (0, 2)
    assert np.array_equal(frames[0].poses, observed.poses)
    assert provenance == {'detection_sha256': 'a' * 64}
    with pytest.raises(FileExistsError):
        save_features(path, [observed], provenance)
    with pytest.raises(ValueError, match='consecutive'):
        save_features(tmp_path / 'wrong.npz', [empty], provenance)
    with np.load(path, allow_pickle=False) as archive:
        legacy = {name: archive[name] for name in archive.files}
    metadata = json.loads(legacy['metadata'].item())
    metadata['schema'] = 'person_detection_features_v1'
    legacy['metadata'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(tmp_path / 'legacy.npz', **legacy)
    with pytest.raises(ValueError, match='Unsupported feature archive schema'):
        load_features(tmp_path / 'legacy.npz')
