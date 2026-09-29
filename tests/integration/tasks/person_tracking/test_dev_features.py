"""All-person extraction retains every row and reuses pose across encoders."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.person_tracking.features import (
    FeatureConfig,
    FeatureExtractor,
    UnpromptedEncoder,
)
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from tests.benchmarks import person_tracking_dev_features as benchmark


class Pose:
    calls = 0
    def predict(self, request: Pose2DFrameSequenceRequest) -> Pose2DResult:
        self.calls += 1
        return Pose2DResult(torch.ones(len(request.bbx_xys), 17, 3))


class Encoder:
    name = 'fixture'
    input_size = (8, 4)
    def embed(self, crops: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1., 0.]]).repeat(len(crops), 1)


def test_empty_frame_many_people_and_saved_pose_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = {'path': str(tmp_path / 'v.mp4'), 'sha256': 'fixture', 'num_frames': 2}
    monkeypatch.setattr(benchmark, 'checked_file', lambda record: Path(record['path']))
    monkeypatch.setattr(benchmark, 'OpenCVVideoFrameReader', lambda *a, **k: [
        SimpleNamespace(index=i, frame=np.full((100, 100, 3), 80, np.uint8)) for i in range(2)])
    boxes = np.tile(np.array([[5., 5., 20., 40.]], np.float32), (8, 1))
    detections = DetectionArchive(np.array([0, 0, 8], np.int64), boxes, np.full(8, .3, np.float32), np.zeros(2, np.float64))
    pose = Pose()
    config = FeatureConfig(appearance_batch_size=3)
    encoder = UnpromptedEncoder(Encoder(), 2)
    frames = benchmark.camera_features(video, detections, FeatureExtractor(pose, encoder, config), config, None)
    other = benchmark.camera_features(video, detections, encoder, config, frames)
    assert pose.calls == 1 and len(frames[0].rows) == 0
    assert other[1].rows.tolist() == list(range(8)) and other[1].appearance_valid.all()
    np.testing.assert_array_equal(other[1].poses, frames[1].poses)
    with pytest.raises(ValueError, match='full saved pose'):
        benchmark.camera_features(video, detections, encoder, config, frames[:1])
    frames[1] = replace(frames[1], boxes=frames[1].boxes.copy())
    frames[1].boxes[0, 0] = 6.
    with pytest.raises(ValueError, match='input detection rows'):
        benchmark.camera_features(video, detections, encoder, config, frames)
