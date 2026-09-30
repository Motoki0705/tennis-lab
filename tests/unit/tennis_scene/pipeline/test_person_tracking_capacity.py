"""Raw person tracking has no player cap and preserves source boxes."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.tasks.person_tracking.sequence import TrackingConfig
from src.tennis_scene.pipeline.components import person_tracking as module
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import SourceVideo


def test_more_than_six_people_keep_their_raw_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Tracker:
        def update(self, detections: Any, frame: Any) -> tuple[np.ndarray, np.ndarray]:
            return np.arange(10, 18, dtype=np.int64), np.arange(7, -1, -1, dtype=np.int64)
    monkeypatch.setattr(module, 'AllPersonAssociator', Tracker)
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', lambda *a, **k: [SimpleNamespace(index=0, frame=np.zeros((10, 10, 3), np.uint8))])
    boxes: np.ndarray = np.arange(32, dtype=np.float32).reshape(8, 4)
    detection = PersonDetectionOutput('cam0', np.array([0, 8], np.int64), boxes, np.full(8, .3, np.float32))
    video = SourceVideo('cam0', tmp_path / 'x.mp4', 'hash', 1, 30., 100, 100)
    result = module.PersonTrackingModule(TrackingConfig(method='all_person_botsort')).process(module.PersonTrackingInput(video, detection))
    assert result.track_ids.tolist() == list(range(10, 18))
    assert result.observed.all() and not result.tracklet_links
    np.testing.assert_array_equal(result.boxes_xyxy[:, 0], boxes[::-1])
