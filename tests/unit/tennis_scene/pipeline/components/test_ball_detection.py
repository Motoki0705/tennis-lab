"""Tests for tennis_scene ball-detection pipeline component."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import src.tennis_scene.pipeline.components.ball_detection as ball_component
from src.tasks.ball_detection.model_io import BallPrediction
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionResult,
)
from src.utils.video import FramePacket
from tests.unit.tennis_scene.pipeline.config_factories import make_ball_config


class _TypedBallPredictor:
    configured_frames = 2

    def predict(self, images: torch.Tensor) -> BallPrediction:
        batch_size = images.shape[0]
        coords = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
        confidence = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
        return BallPrediction(
            coords=coords.repeat(batch_size, 1, 1),
            confidence=confidence.repeat(batch_size, 1),
            heatmaps=torch.zeros((batch_size, 2, 2, 3)),
        )


def test_trajectory_gate_zeroes_rejected_pipeline_frames(tmp_path) -> None:
    frame: NDArray[np.float32] = np.arange(12, dtype=np.float32)
    ball_uv_px = np.stack(
        [
            50.0 + 20.0 * frame,
            np.full(12, 120.0, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    ball_uv_px[6, 0] += 180.0
    ball_uv = ball_uv_px.copy()
    ball_uv[:, 0] /= 639.0
    ball_uv[:, 1] /= 359.0
    result = BallDetectionResult(
        ball_uv=ball_uv[np.newaxis, ...],
        ball_uv_px=ball_uv_px[np.newaxis, ...],
        visibility=np.ones((1, 12), dtype=np.bool_),
        score=np.full((1, 12), 0.9, dtype=np.float32),
    )
    module = BallDetectionModule(make_ball_config(tmp_path))

    gated = module._apply_trajectory_gate(result)

    assert not bool(gated.visibility[0, 6])
    np.testing.assert_array_equal(gated.ball_uv[0, 6], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(
        gated.ball_uv_px[0, 6],
        np.zeros(2, dtype=np.float32),
    )
    assert gated.score[0, 6] == 0.0
    is_valid, errors = gated.validate()
    assert is_valid
    assert errors == []


def test_predict_video_consumes_typed_task_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packets: list[FramePacket] = [
        FramePacket(
            index=index,
            frame=np.zeros((4, 6, 3), dtype=np.uint8),
            original_size=(6, 4),
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        ball_component,
        "OpenCVVideoFrameReader",
        lambda _path, *, max_frames: packets[:max_frames],
    )
    config = replace(make_ball_config(tmp_path), image_size=(4, 6))
    module = BallDetectionModule(config)
    module._pipeline = _TypedBallPredictor()  # type: ignore[assignment]

    coords, confidence = module._predict_video(Path("unused.mp4"), max_frames=2)

    np.testing.assert_allclose(coords, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(confidence, [0.6, 0.8])
