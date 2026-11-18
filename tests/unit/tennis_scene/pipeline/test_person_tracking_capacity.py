"""Raw person tracking has no player cap and preserves source boxes."""
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.person_tracking.sequence import TrackingConfig
from src.tennis_scene.pipeline.components import person_tracking as module
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import SourceVideo


def test_more_than_six_people_keep_their_raw_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Link:
        def __init__(self, path: Path) -> None:
            pass
        def links(self, boxes: np.ndarray, observed: np.ndarray) -> tuple[dict[int, int], list[Any]]:
            return {i: i for i in range(len(boxes))}, []
    monkeypatch.setattr(module, 'AFLink', Link)
    boxes = np.array([[100 * i + 10, 20, 100 * i + 50, 120] for i in range(8)], np.float32)
    detection = PersonDetectionOutput('cam0', np.arange(0, 33, 8, dtype=np.int64), np.tile(boxes, (4, 1)),
        np.full(32, .9, np.float32), np.arange(32, dtype=np.int64))
    video = SourceVideo('cam0', tmp_path / 'x.mp4', 'hash', 4, 30., 800, 200)
    from src.tennis_scene.pipeline.components.person_features import (
        PersonAppearanceFeaturesOutput,
        PersonPoseFeaturesOutput,
    )
    poses: np.ndarray = np.zeros((32, 17, 3), np.float32)
    poses[..., :2] = detection.boxes_xyxy[:, None, :2] + 10
    poses[..., 2] = 1
    embeddings: np.ndarray = np.zeros((32, 1280), np.float32)
    embeddings[:, :8] = np.tile(np.eye(8, dtype=np.float32), (4, 1))
    pose_features = PersonPoseFeaturesOutput('cam0', detection.frame_offsets, detection.source_rows, poses)
    appearance = PersonAppearanceFeaturesOutput('cam0', detection.frame_offsets, detection.source_rows, embeddings, np.ones(32, bool), TrackingConfig().encoder, "a" * 64)
    tracker = module.PersonTrackingModule(TrackingConfig(), aflink_checkpoint=tmp_path / 'aflink.pth')
    result = tracker.process(module.PersonTrackingInput(video, detection, pose_features, appearance))
    assert result.track_ids.tolist() == list(range(1, 9))
    assert result.observed[:, 2:].all() and not result.tracklet_links
    assert result.evidence is not None
    np.testing.assert_array_equal(result.evidence.detection_rows[:, 3], np.arange(24, 32))
    np.testing.assert_array_equal(result.boxes_xyxy[:, 3], boxes)
