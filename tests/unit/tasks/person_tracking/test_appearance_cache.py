from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.tasks.person_tracking.appearance_cache import cached_appearance
from src.tasks.player_association.association.associate import CameraTracks
from src.utils.geometry.triangulation import PinholeCamera


class Encoder:
    name = 'test'
    input_size = (16, 8)
    calls = 0

    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.nn.functional.normalize(crops.mean((2, 3)) + .1, dim=-1)


def test_exact_crop_cache_reuses_only_same_input_and_checks_stored_content(tmp_path: Path) -> None:
    camera = PinholeCamera('cam0', np.eye(3), np.eye(3), np.array([0., 0., 10.]))
    boxes = np.array([[[10., 10., 30., 80.]], [[50., 10., 70., 80.]]], np.float32)
    tracks = CameraTracks(camera, (100, 100), np.array([4, 7]), boxes, np.ones((2, 1), bool))
    video = {'path': 'video.mp4', 'sha256': 'video-hash'}
    encoder = Encoder()
    frame: np.ndarray = np.zeros((100, 100, 3), np.uint8)
    frame[:, 40:] = 255
    with patch('src.tasks.person_tracking.appearance_cache.OpenCVVideoFrameReader', return_value=[SimpleNamespace(index=0, frame=frame)]):
        first, stats = cached_appearance(tracks, np.array([[True], [False]]), video, encoder, 'weights', tmp_path)
        assert stats['new_crops'] == 1 and encoder.calls == 1
        assert len(first[0].frames) == 1 and len(first[1].frames) == 0
        second, stats = cached_appearance(tracks, np.array([[True], [True]]), video, encoder, 'weights', tmp_path)
        assert stats['new_crops'] == stats['cached_crops'] == 1 and encoder.calls == 2
        np.testing.assert_array_equal(first[0].embeddings, second[0].embeddings)
        cached_appearance(tracks, np.array([[True], [False]]), video, encoder, 'different-weights', tmp_path)
        assert encoder.calls == 3
        for path in tmp_path.glob('*.npz'):
            with np.load(path) as a:
                sha, vector = a['sha256'], a['embedding'].copy()
            vector[0] += 1
            np.savez_compressed(path, embedding=vector, sha256=sha)
        with pytest.raises(ValueError, match='hash mismatch'):
            cached_appearance(tracks, np.array([[True], [False]]), video, encoder, 'weights', tmp_path)
