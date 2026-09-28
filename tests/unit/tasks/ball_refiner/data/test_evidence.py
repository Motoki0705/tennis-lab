"""Source units, window selection, weak peaks and cache corruption boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallPrediction,
)
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.evidence_inference import infer_clip_evidence
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


class PatternPredictor:
    configured_frames = 4

    def __init__(self):
        self.windows_seen = 0

    def predict(self, images, *, candidate_config):
        assert images.dtype == torch.float32
        assert 0 <= images.min() <= images.max() <= 1
        b, t = images.shape[:2]
        maps = torch.zeros(b, t, 7, 9)
        for i in range(b):
            for j in range(t):
                # Increasing window score would choose the LATER window in an overlap.
                maps[i, j, 4, 5] = 0.2 + 0.01 * (self.windows_seen + i) + 0.001 * j
                maps[i, j, 0, 0] = 1e-7  # weak boundary peak must survive
        self.windows_seen += b
        peaks = decode_candidates(maps, config=candidate_config, subpixel_refine=False)
        return BallPrediction(peaks.coords[..., 0, :], peaks.scores[..., 0], maps, peaks)


def make_store(tmp_path, n=11):
    write_store_clip(tmp_path, "tracknet/game1/clip1", [frame(1000 + 2 * i, ball()) for i in range(n)])
    metadata = tmp_path / "metadata.json"
    data = json.loads(metadata.read_text())
    data["clips"][0].update(source_width=128, source_height=96)
    metadata.write_text(json.dumps(data))
    return BallFrameStore(tmp_path)


def infer(store, *, stride=2, batch_size=2):
    return infer_clip_evidence(
        store, store.clips[0], PatternPredictor(), image_size_hw=(24, 32),
        stride=stride, batch_size=batch_size,
        config=BallCandidateConfig(max_candidates=3, nms_kernel=3, patch_size=3),
    )


def test_no_score_selection_and_every_field_comes_from_same_window(tmp_path):
    store = make_store(tmp_path)
    evidence = infer(store)
    # Frame 8 ties between starts 6 and 7 and keeps the earlier start 6.
    assert evidence.window_start.tolist() == [0, 0, 0, 2, 2, 4, 4, 6, 6, 7, 7]
    np.testing.assert_array_equal(evidence.window_start + evidence.time_index, np.arange(11))
    expected_score = np.array([
        0.2 + 0.01 * [0, 2, 4, 6, 7].index(int(start)) + 0.001 * slot
        for start, slot in zip(evidence.window_start, evidence.time_index, strict=True)
    ], np.float32)
    np.testing.assert_allclose(evidence.argmax_score, expected_score)
    np.testing.assert_array_equal(evidence.candidates.scores[0, :, 0], evidence.argmax_score)
    np.testing.assert_array_equal(evidence.candidates.patches[0, :, 0, 1, 1], evidence.argmax_score)
    np.testing.assert_allclose(evidence.argmax_uv, np.tile([5 / 8 * 126 / 127, 4 / 6 * 94 / 95], (11, 1)))
    np.testing.assert_allclose(evidence.timestamps_seconds, np.arange(11) * 2 / 30)
    assert evidence.pts[0] == 1000
    assert evidence.candidates.valid[0, :, :2].all()
    assert not evidence.candidates.valid[0, :, 2].any()
    assert (evidence.candidates.scores[0, :, 1] == 1e-7).all()
    assert evidence.candidates.patch_valid[0, 0, 1].tolist() == [
        [False, False, False], [False, True, True], [False, True, True],
    ]
    np.testing.assert_array_equal(evidence.candidates.cells[0, 0], [[5, 4], [0, 0], [0, 0]])


@pytest.mark.parametrize("n", [4, 5, 8, 11])
def test_backfilled_tail_is_unique_and_batch_partition_does_not_change_evidence(tmp_path, n):
    store = make_store(tmp_path, n)
    first, second = infer(store, stride=3, batch_size=1), infer(store, stride=3, batch_size=3)
    for name, array in first.arrays().items():
        np.testing.assert_array_equal(array, second.arrays()[name])
    assert first.window_start[-1] == n - 4
    assert first.time_index[-1] == 3


@pytest.mark.parametrize("n", [1, 2, 3])
def test_short_clips_are_rejected_instead_of_repeating_frames(tmp_path, n):
    with pytest.raises(ValueError, match="padding is forbidden"):
        infer(make_store(tmp_path, n))


def test_bad_stride_fails_before_inference(tmp_path):
    with pytest.raises(ValueError, match="stride"):
        infer(make_store(tmp_path), stride=5)


@pytest.mark.parametrize("field,change", [
    ("window_start", lambda array: array.__setitem__(0, -1)),
    ("pts", lambda array: array.__setitem__(1, array[0])),
    ("candidate_patches", lambda array: array.__setitem__((0, 0, 1, 1), 0.9)),
    ("candidate_patch_valid", lambda array: array.__setitem__((0, 1, 0, 0), True)),
    ("candidate_coords", lambda array: array.__setitem__((0, 0, 0), np.nan)),
    ("candidate_cells", lambda array: array.__setitem__((0, 0, 0), 500)),
])
def test_corrupt_cache_arrays_fail_validation(tmp_path, field, change):
    evidence = infer(make_store(tmp_path))
    arrays = {name: value.copy() for name, value in evidence.arrays().items()}
    change(arrays[field])
    with pytest.raises(ValueError):
        ClipEvidence.from_arrays(arrays, config=evidence.candidates.config,
                                 heatmap_size_hw=evidence.heatmap_size_hw, window_length=4)


def test_flat_heatmaps_are_generated_evidence_with_zero_candidates(tmp_path):
    class FlatPredictor:
        configured_frames = 4

        def predict(self, images, *, candidate_config):
            maps = torch.full((*images.shape[:2], 7, 9), 0.3)
            return BallPrediction(
                torch.zeros(*images.shape[:2], 2), torch.full(images.shape[:2], 0.3), maps,
                decode_candidates(maps, config=candidate_config, subpixel_refine=False),
            )

    store = make_store(tmp_path, 4)
    evidence = infer_clip_evidence(
        store, store.clips[0], FlatPredictor(), image_size_hw=(24, 32),
        stride=4, batch_size=1, config=BallCandidateConfig(),
    )
    assert not evidence.candidates.valid.any()
    assert not evidence.candidates.patch_valid.any()
    restored = ClipEvidence.from_arrays(
        evidence.arrays(), config=evidence.candidates.config, heatmap_size_hw=(7, 9), window_length=4,
    )
    np.testing.assert_array_equal(restored.argmax_score, evidence.argmax_score)
    with pytest.raises(ValueError, match="does not belong"):
        infer_clip_evidence(store, replace(store.clips[0], source_width=256), FlatPredictor(),
                            image_size_hw=(24, 32), stride=4, batch_size=1, config=BallCandidateConfig())
