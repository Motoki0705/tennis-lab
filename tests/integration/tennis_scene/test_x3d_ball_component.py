"""Real short-video reader, native resize, component masks and artifact storage."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from weakref import ref

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import BallPrediction
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionInput,
    BallDetectionModule,
    BallDetectionOutput,
)
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.storage.codec import (
    encode_value,
    restore_type,
    unpack_value,
)
from src.utils.data.heatmaps import heatmaps_to_argmax
from src.utils.video import VideoWriter
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_x3d_ball_config,
    materialize_model_weights,
)


def test_short_video_keeps_native_grid_evidence_and_presence_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "short.mp4"
    with VideoWriter(video, fps=30.) as writer:
        for _ in range(33):
            writer.write_frame(np.tile(np.array([220, 80, 20], dtype=np.uint8), (24, 40, 1)))
    calls = []

    class Predictor:
        configured_frames = 32
        def predict(self, images, *, candidate_config):
            assert images.shape == (1, 32, 3, 720, 1280) and images.dtype == torch.uint8
            assert int(images[0, 0, 0, 0, 0]) > int(images[0, 0, 2, 0, 0]) + 150
            calls.append(len(images))
            maps = torch.zeros(1, 32, 180, 320)
            maps[:, :, 90, 160] = .75
            maps[:, 0, 90, 160] = .25  # absent single point, retained in evidence
            uv, score = heatmaps_to_argmax(maps)
            return BallPrediction(uv, score, maps, decode_candidates(maps, config=candidate_config, subpixel_refine=False))

    base = make_x3d_ball_config(tmp_path)
    module = BallDetectionModule(replace(base, batch_size=1, score_threshold=.5, pin_memory=False,
                                         trajectory_gate=replace(base.trajectory_gate, enabled=False)))
    materialize_model_weights(module)
    predictors = []
    def load():
        predictor = Predictor()
        predictors.append(ref(predictor))
        return predictor
    monkeypatch.setattr(module, "_load_predictor", load)
    out = module.process(BallDetectionInput(SourceVideo("cam", video, "fixture", 33, 30., 40, 24)))
    assert calls == [1, 1] and all(predictor() is None for predictor in predictors)
    assert out.observed.sum() == 32 and not out.observed[0]
    np.testing.assert_allclose(out.uv_px[1], [160 / 319 * 39, 90 / 179 * 23])
    assert out.evidence is not None
    assert out.evidence.heatmaps.shape == (33, 180, 320)
    assert out.evidence.candidate_scores[0, 0] == .25
    np.testing.assert_array_equal(out.evidence.selected_window_start, [0] * 17 + [1] * 16)
    arrays: dict[str, Any] = {}
    encoded = encode_value(out, tmp_path, arrays)
    restored = restore_type(unpack_value(encoded, tmp_path, arrays), BallDetectionOutput)
    np.testing.assert_array_equal(restored.observed, out.observed)
    np.testing.assert_array_equal(restored.evidence.candidate_uv_px, out.evidence.candidate_uv_px)
