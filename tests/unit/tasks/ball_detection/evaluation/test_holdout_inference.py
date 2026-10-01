"""Whole-frame coverage and selected-window provenance through real JPEG storage."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.evaluation.holdout_inference import predict_store_clip
from src.tasks.ball_detection.evaluation.holdout_metrics import clip_references
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallCandidates,
    BallPrediction,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


class WindowPredictor:
    configured_frames = 4

    def __init__(self, ties=True):
        self.ties = ties

    def predict(self, images, *, candidate_config):
        assert images.dtype == torch.float32
        assert images.min() >= 0 and images.max() <= 1
        b, t = images.shape[:2]
        k, p = candidate_config.max_candidates, candidate_config.patch_size
        coords = torch.arange(t, dtype=torch.float32)[None, :, None].expand(b, t, 2) / 10
        score = torch.full((b, t), 0.8) if self.ties else 0.5 + coords[..., 0]
        return BallPrediction(coords, score, torch.zeros(b, t, 3, 3), BallCandidates(
            coords[:, :, None].expand(b, t, k, 2), score[:, :, None].expand(b, t, k),
            torch.ones(b, t, k, dtype=torch.bool), torch.zeros(b, t, k, 2, dtype=torch.long),
            torch.zeros(b, t, k, p, p), torch.ones(b, t, k, p, p, dtype=torch.bool), candidate_config,
        ))


def make_clip(tmp_path, n):
    write_store_clip(tmp_path, "meiji/video_001/clip_000/cam0", [frame(i, ball()) for i in range(n)], source="meiji", split="test")
    store = BallFrameStore(tmp_path)
    clip = replace(store.clips[0], camera_id="cam0", source_width=128, source_height=96)
    return store, clip


@pytest.mark.parametrize("n", [1, 2, 4, 8, 11])
def test_tail_and_short_clip_cover_unique_source_frames(tmp_path, n):
    store, clip = make_clip(tmp_path, n)
    result = predict_store_clip(store, clip, WindowPredictor(), image_size=(24, 32), stride=3,
                                batch_size=2, candidates=BallCandidateConfig())
    assert len(result.uv) == n
    assert (result.window_start >= 0).all()
    assert result.time_index[-1] == 3
    if n >= 4:
        np.testing.assert_array_equal(result.window_start + result.time_index, np.arange(n))
    # Last padded slot wins a score tie, but is still only one original frame.
    np.testing.assert_allclose(result.uv[-1], np.asarray([126, 94]) * 0.3, rtol=1e-6)
    np.testing.assert_allclose(result.candidate_uv[-1, 0], result.uv[-1])
    ref = clip_references(store, clip)
    np.testing.assert_allclose(ref.uv, np.tile([20, 40], (n, 1)))


def test_overlap_max_score_keeps_all_candidate_evidence_from_same_window(tmp_path):
    store, clip = make_clip(tmp_path, 11)
    common = dict(image_size=(24, 32), stride=3, batch_size=3, candidates=BallCandidateConfig())
    tied = predict_store_clip(store, clip, WindowPredictor(), **common)
    ranked = predict_store_clip(store, clip, WindowPredictor(ties=False), **common)
    assert tied.window_start[3] == 3
    assert ranked.window_start[3] == 0
    assert ranked.time_index[3] == 3
    np.testing.assert_allclose(ranked.candidate_uv[:, 0], ranked.uv)
    np.testing.assert_allclose(ranked.candidate_score[:, 0], ranked.score)


def test_holes_and_multiple_ground_truth_instances_are_rejected(tmp_path):
    store, clip = make_clip(tmp_path, 8)
    with pytest.raises(ValueError, match="Stride"):
        predict_store_clip(store, clip, WindowPredictor(), image_size=(24, 32), stride=5,
                           batch_size=2, candidates=BallCandidateConfig())
    write_store_clip(tmp_path, clip.clip_id, [frame(0, ball(), ball(track="b002"))], source="meiji", split="test")
    store = BallFrameStore(tmp_path)
    with pytest.raises(ValueError, match="multiple references"):
        clip_references(store, replace(store.clips[0], camera_id="cam0"))
