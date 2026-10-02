"""Split isolation and explicit cache projection, without labels or a GPU."""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import FeatureConfig, encode_appearance
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    spec = importlib.util.spec_from_file_location('recalibration_features', PROJECT_ROOT / 'tests/benchmarks/association_recalibration_features.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_selection_ignores_outcomes_and_excludes_dev_reserved_and_labels(benchmark: ModuleType) -> None:
    metadata = {f'video_{v:03}/clip_{c:03}': {'num_frames': 100 + 100 * c, 'fps': 50., 'status': 'failed' if c == 4 else 'ok'}
                for v in range(3) for c in range(7)}
    reserved = {'video_000/clip_001', 'video_001/clip_002', 'video_002/clip_000'}
    labelled = {'video_000/clip_002', 'video_002/clip_001'}
    selected = benchmark.select_clips(metadata, reserved, labelled)
    assert selected == ['video_000/clip_003', 'video_000/clip_004',
                        'video_001/clip_000', 'video_001/clip_003',
                        'video_002/clip_002', 'video_002/clip_003']
    for row in metadata.values():
        row['status'] = 'different'
    assert benchmark.select_clips(metadata, reserved, labelled) == selected
    with pytest.raises(ValueError, match='fewer than two'):
        benchmark.select_clips(metadata, reserved | set(selected), labelled | {'video_000/clip_005', 'video_000/clip_006'})


def test_overlap_guard_checks_original_camera_frames(benchmark: ModuleType) -> None:
    def meta(start: int, end: int) -> dict[str, object]:
        return {'cameras': [{'source_path': 'raw/cam1.mp4', 'source_frame_start': start, 'source_frame_end': end}]}
    benchmark.assert_disjoint({'fit': meta(10, 20)}, {'test': meta(21, 30)})
    with pytest.raises(ValueError, match='Overlapping'):
        benchmark.assert_disjoint({'fit': meta(10, 20)}, {'test': meta(20, 30)})


def test_cache_projection_matches_actual_extractor_masks_without_inventing_features(benchmark: ModuleType) -> None:
    import torch

    class Encoder:
        name = 'test'
        dimension = 2
        input_size = (16, 8)

        def embed(self, crops: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
            return torch.tensor([1., 0.]).expand(len(crops), 2)

    # Rounding, clipped borders, exact threshold, and boxes completely outside the frame.
    boxes = np.array([[10, 1, 30, 16.49], [10, 1, 30, 16.5], [10, 1070, 30, 1090],
                      [-20, 1, -10, 40], [10, 1, 30, 17]], np.float32)
    rows: np.ndarray = np.arange(5, dtype=np.int64)
    scores: np.ndarray = np.ones(5, np.float32)
    pose: np.ndarray = np.zeros((5, 17, 3), np.float32)
    image: np.ndarray = np.zeros((1080, 1920, 3), np.uint8)
    before, after = FeatureConfig(min_appearance_height_px=1., appearance_batch_size=8), FeatureConfig()
    cached = encode_appearance(0, image, rows, boxes, scores, pose, Encoder(), before)
    actual = encode_appearance(0, image, rows, boxes, scores, pose, Encoder(), after)
    projected, changed = benchmark.restrict_appearance([cached], before, after, (1920, 1080))
    assert changed == 3
    for field in ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid'):
        np.testing.assert_array_equal(getattr(projected[0], field), getattr(actual, field))
    # The original archive is not mutated; values newly marked absent are zero only in the projection.
    assert cached.appearance_valid.sum() == 4
    with pytest.raises(ValueError, match='incompatible'):
        benchmark.restrict_appearance(projected, after, before, (1920, 1080))
    with pytest.raises(ValueError, match='incompatible'):
        benchmark.restrict_appearance([cached], before, replace(after, bbox_enlarge=1.3), (1920, 1080))


def test_checked_input_refuses_stale_identity(benchmark: ModuleType, tmp_path: Path) -> None:
    from src.utils.checksum import dual_sha256
    file = tmp_path / 'feature.npz'
    file.write_bytes(b'original')
    record = {'path': str(file), 'sha256': dual_sha256(file)}
    assert benchmark.checked(record) == file
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        benchmark.checked(record)


def test_empty_frame_projection_keeps_timeline(benchmark: ModuleType) -> None:
    frame = DetectionFeatures(12, np.empty(0, np.int64), np.zeros((0, 4), np.float32), np.empty(0, np.float32),
                              np.zeros((0, 17, 3), np.float32), np.zeros((0, 2), np.float32), np.empty(0, bool))
    restored, changed = benchmark.restrict_appearance([frame], FeatureConfig(), FeatureConfig(), (1920, 1080))
    assert restored[0].frame == 12 and changed == 0
