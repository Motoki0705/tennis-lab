"""Tests for tennis-scene 2D ball trajectory clip preparation."""

from __future__ import annotations

import numpy as np
import pytest

from src.tennis_scene.pipeline.components.ball_trajectory import (
    complete_ball_trajectory_clip,
)


def test_complete_ball_trajectory_clip_interpolates_detector_gaps() -> None:
    ball_uv: np.ndarray = np.zeros((1, 5, 2), dtype=np.float32)
    ball_uv[0, 1] = [0.2, 0.4]
    ball_uv[0, 4] = [0.8, 1.0]
    visibility = np.array([[False, True, False, False, True]], dtype=np.bool_)

    result = complete_ball_trajectory_clip(ball_uv, visibility)

    assert result.start_frame == 1
    assert result.end_frame == 5
    np.testing.assert_array_equal(
        result.ball_mask,
        np.array([[False, True, True, True, True]], dtype=np.bool_),
    )
    np.testing.assert_allclose(
        result.ball_uv[0],
        np.array(
            [
                [0.0, 0.0],
                [0.2, 0.4],
                [0.4, 0.6],
                [0.6, 0.8],
                [0.8, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_complete_ball_trajectory_clip_rejects_no_observations() -> None:
    with pytest.raises(ValueError, match="no 2D observations"):
        complete_ball_trajectory_clip(
            np.zeros((1, 3, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.bool_),
        )


def test_complete_ball_trajectory_clip_rejects_camera_without_clip_observation() -> None:
    ball_uv: np.ndarray = np.zeros((2, 3, 2), dtype=np.float32)
    ball_uv[0, 1] = [0.5, 0.5]
    visibility = np.array(
        [
            [False, True, False],
            [False, False, False],
        ],
        dtype=np.bool_,
    )

    with pytest.raises(ValueError, match="camera 1 has no finite 2D ball observations"):
        complete_ball_trajectory_clip(ball_uv, visibility)
