"""Raw person tracking has no player cap and preserves source boxes."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.sequence import TrackingConfig
from src.tennis_scene.pipeline.components import person_tracking as module
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config


def test_more_than_six_people_keep_their_raw_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Pose:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def unload(self) -> None:
            pass
    class Encoder:
        name = 'clipreid_vitb16_market1501'
        input_size = (256, 128)
        def embed(self, crops: torch.Tensor) -> torch.Tensor:
            raise AssertionError('Features are supplied by the fixture')
    class Features:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def extract(self, frame: int, image: np.ndarray, rows: np.ndarray,
                    boxes: np.ndarray, scores: np.ndarray) -> DetectionFeatures:
            poses: np.ndarray = np.zeros((8, 17, 3), np.float32)
            poses[..., :2] = boxes[:, None, :2] + 10
            poses[..., 2] = 1
            return DetectionFeatures(frame, rows, boxes, scores, poses, np.eye(8, dtype=np.float32), np.ones(8, bool))
    class Link:
        def __init__(self, path: Path) -> None:
            pass
        def links(self, boxes: np.ndarray, observed: np.ndarray) -> tuple[dict[int, int], list[Any]]:
            return {i: i for i in range(len(boxes))}, []
    monkeypatch.setattr(module, 'ViTPosePose2D', Pose)
    monkeypatch.setattr(module, 'FeatureExtractor', Features)
    monkeypatch.setattr(module, 'AFLink', Link)
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', lambda *a, **k:
        [SimpleNamespace(index=i, frame=np.zeros((200, 800, 3), np.uint8)) for i in range(4)])
    boxes = np.array([[100 * i + 10, 20, 100 * i + 50, 120] for i in range(8)], np.float32)
    detection = PersonDetectionOutput('cam0', np.arange(0, 33, 8, dtype=np.int64), np.tile(boxes, (4, 1)),
        np.full(32, .9, np.float32), np.arange(32, dtype=np.int64))
    video = SourceVideo('cam0', tmp_path / 'x.mp4', 'hash', 4, 30., 800, 200)
    tracker = module.PersonTrackingModule(TrackingConfig(), people=make_people_config(tmp_path),
        encoder=Encoder, aflink_checkpoint=tmp_path / 'aflink.pth')
    result = tracker.process(module.PersonTrackingInput(video, detection))
    assert result.track_ids.tolist() == list(range(1, 9))
    assert result.observed[:, 2:].all() and not result.tracklet_links
    assert result.evidence is not None
    np.testing.assert_array_equal(result.evidence.detection_rows[:, 3], np.arange(24, 32))
    np.testing.assert_array_equal(result.boxes_xyxy[:, 3], boxes)
