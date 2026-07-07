"""Tests for tennis_scene ball-detection pipeline component."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionConfig,
    BallDetectionModule,
    BallDetectionResult,
)


def test_trajectory_gate_zeroes_rejected_pipeline_frames() -> None:
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
    module = BallDetectionModule(
        BallDetectionConfig(
            checkpoint="dummy.ckpt",
            device="cpu",
            trajectory_gate=TrajectoryGateConfig(enabled=True),
        )
    )

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
