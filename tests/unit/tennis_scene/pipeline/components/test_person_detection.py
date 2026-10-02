"""Full-frame detections do not depend on calibration or a court polygon."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.submodules.models import PersonDetectionResult
from src.tennis_scene.pipeline.components import person_detection as module
from src.tennis_scene.pipeline.contracts import SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config


def test_full_frame_keeps_off_court_people_without_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    boxes = np.array([[1, 1, 20, 40], [600, 400, 630, 470]], np.float32)
    calls: list[dict[str, Any]] = []

    class Detector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(kwargs)
        def predict(self, request: Any) -> PersonDetectionResult:
            return PersonDetectionResult(boxes, np.array([.31, .95], np.float32))
        def unload(self) -> None:
            pass

    monkeypatch.setattr(module, 'DinoPersonDetector', Detector)
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', lambda *a, **k: [SimpleNamespace(index=0, frame=np.zeros((480, 640, 3), np.uint8))])
    video = SourceVideo('cam2', tmp_path / 'x.mp4', 'hash', 1, 30., 640, 480)
    result = module.PersonDetectionModule(make_people_config(tmp_path)).process(module.PersonDetectionInput(video))
    assert result.frame_offsets.tolist() == [0, 2]
    np.testing.assert_array_equal(result.boxes_xyxy, boxes)
    assert calls and not module.PersonDetectionModule.io.inputs


def test_optional_merge_records_original_clip_rows_and_roundtrips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
    boxes = np.array([[0, 0, 10, 10], [0, 0, 8, 10]], np.float32)
    class Detector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def predict(self, request: Any) -> PersonDetectionResult:
            return PersonDetectionResult(boxes, np.array([.7, .9], np.float32))
        def unload(self) -> None:
            pass
    monkeypatch.setattr(module, 'DinoPersonDetector', Detector)
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', lambda *a, **k:
        [SimpleNamespace(index=i, frame=np.zeros((480, 640, 3), np.uint8)) for i in range(2)])
    video = SourceVideo('cam0', tmp_path / 'x.mp4', 'hash', 2, 30., 640, 480)
    inputs = module.PersonDetectionInput(video)
    result = module.PersonDetectionModule(make_people_config(tmp_path), merge_duplicates=True).process(inputs)
    assert result.source_rows is not None and result.source_rows.tolist() == [1, 3]
    assert result.frame_offsets.tolist() == [0, 1, 2]
    assert [(m.kept_row, m.dropped_row, m.iou) for m in result.duplicate_merges] == [(1, 0, .8), (3, 2, .8)]
    codec = ArtifactCodec(type(result))
    payload, arrays = codec.dump(result, tmp_path)
    loaded = codec.load(payload, tmp_path, arrays)
    assert loaded.duplicate_merges == result.duplicate_merges
    original = module.PersonDetectionModule(make_people_config(tmp_path)).process(inputs)
    assert original.source_rows is not None and original.source_rows.tolist() == [0, 1, 2, 3]
    assert not original.duplicate_merges


def test_disabled_detection_does_not_load_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('disabled detector')
    monkeypatch.setattr(module, 'DinoPersonDetector', forbidden)
    video = SourceVideo('cam2', tmp_path / 'x.mp4', 'hash', 5, 30., 640, 480)
    result = module.PersonDetectionModule(make_people_config(tmp_path), enabled=False).process(module.PersonDetectionInput(video))
    assert result.frame_offsets.tolist() == [0] * 6
