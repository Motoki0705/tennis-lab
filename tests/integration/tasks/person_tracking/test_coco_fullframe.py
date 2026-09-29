"""The queue entry point keeps low-score, out-of-ROI detections and all frames."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.submodules.models import PersonDetectionResult
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def test_fullframe_job_does_not_apply_roi_or_an_extra_score_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    module = importlib.import_module('person_coco_fullframe')
    raw = importlib.import_module('player_detection_far_diagnosis')
    video = tmp_path / 'video'
    video.write_bytes(b'input')
    weight = tmp_path / 'weight'
    weight.write_bytes(b'weight')
    monkeypatch.setattr(raw, 'OpenCVVideoFrameReader', lambda *a, **kw: [
        SimpleNamespace(frame=np.zeros((10, 10, 3), np.uint8)) for _ in range(2)])
    boxes = np.array([[1, 1, 3, 3], [7, 7, 9, 9]], np.float32)
    monkeypatch.setattr(raw, 'predict_timed', lambda *a: (
        PersonDetectionResult(boxes, np.array([.02, .01], np.float32)), 1.))
    class Detector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs['confidence'] == .01
            assert (kwargs['short_side'], kwargs['max_long_side']) == (800, 1333)

        def load(self) -> None:
            pass

        def unload(self) -> None:
            pass

    monkeypatch.setattr(module, 'DinoPersonDetector', Detector)
    monkeypatch.setattr(module.torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(total_memory=16 * 1024**3))
    monkeypatch.setattr(module.torch.cuda, 'set_per_process_memory_fraction', lambda _: None)
    monkeypatch.setattr(module.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(module.torch.cuda, 'max_memory_allocated', lambda: 1)
    monkeypatch.setattr(module.torch.cuda, 'max_memory_reserved', lambda: 2)
    plan: dict[str, Any] = {'torch_allocator_limit_bytes': 6 * 1024**3, 'repository': 'dummy', 'archives': {},
        'weights': {'coco': {'path': str(weight), 'sha256': dual_sha256(weight)}}, 'inputs': [{
            'clip': 'video_000/clip_000', 'camera': 'cam0', 'roi': [[0, 0], [4, 0], [4, 4]],
            'video': {'path': str(video), 'sha256': dual_sha256(video), 'num_frames': 2}}]}
    module.infer(plan, tmp_path)
    saved = plan['archives']['coco_fullframe_0.01']['video_000/clip_000/cam0']
    archive = DetectionArchive.load(saved)
    assert archive.offsets.tolist() == [0, 2, 4]
    np.testing.assert_allclose(archive.at(1).boxes_xyxy, boxes)
    assert (tmp_path / 'inference.json').exists()
    with pytest.raises(FileExistsError):
        module.infer(plan, tmp_path)
