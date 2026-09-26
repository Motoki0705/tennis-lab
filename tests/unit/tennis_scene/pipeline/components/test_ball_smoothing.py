"""The three smoothers preserve flight events, support, and measurable trajectory shape."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_smoothing import (
    BallSmoothingConfig,
    BallSmoothingMethod,
    ball_event_frames,
    smooth_ball_positions,
)


def _two_flights() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fps = 60.0
    frames = np.arange(81)
    local_time = np.where(frames <= 40, frames, frames - 40) / fps
    truth = np.column_stack((
        .3 * np.sin(frames / 24),
        np.where(frames <= 40, frames, 80 - frames) * .17,
        .12 + .5 * 9.81 * (40 / fps) * local_time - .5 * 9.81 * local_time**2,
    )).astype(np.float32)
    noise = np.random.default_rng(47).normal(0, .027, truth.shape).astype(np.float32)
    observed = truth + noise
    observed[20] += np.array([.42, -.35, .48], np.float32)
    valid: NDArray[np.bool_] = np.ones(len(frames), bool)
    valid[61:63] = False
    observed[~valid] = 0
    return truth, observed, valid


@pytest.mark.parametrize("method", ["savgol", "robust_spline", "ballistic_rts"])
def test_smoothers_reduce_noise_without_blurring_bounce_or_filling_gaps(method: BallSmoothingMethod) -> None:
    truth, observed, valid = _two_flights()
    config = BallSmoothingConfig(method=method)
    result, events = smooth_ball_positions(observed, valid, fps=60., config=config)
    assert any(abs(int(event) - 40) <= 2 for event in events)
    np.testing.assert_array_equal(result[events], observed[events])
    np.testing.assert_array_equal(result[~valid], 0)
    assert result.dtype == np.float32
    before = np.sqrt(np.mean((observed[valid] - truth[valid]) ** 2))
    after = np.sqrt(np.mean((result[valid] - truth[valid]) ** 2))
    assert after < before * .75


def test_no_smoothing_is_an_exact_copy_and_invalid_input_is_rejected() -> None:
    _, observed, valid = _two_flights()
    result, events = smooth_ball_positions(observed, valid, fps=60., config=BallSmoothingConfig())
    np.testing.assert_array_equal(result, observed)
    assert not np.shares_memory(result, observed)
    assert len(events) == 0
    with pytest.raises(ValueError, match="zero outside"):
        smooth_ball_positions(replace_array(observed, 61, np.array([1., 0., 0.], np.float32)), valid,
            fps=60., config=BallSmoothingConfig(method="savgol"))
    with pytest.raises(ValueError, match="window_frames"):
        replace(BallSmoothingConfig(), window_frames=8)


def replace_array(values: np.ndarray, index: int, row: np.ndarray) -> np.ndarray:
    copy = values.copy()
    copy[index] = row
    return copy


def test_event_detector_does_not_bridge_missing_frames() -> None:
    _, observed, valid = _two_flights()
    events = ball_event_frames(observed, valid, BallSmoothingConfig())
    assert not np.isin([61, 62], events).any()
    assert np.all(valid[events])


def test_smoothing_cannot_reintroduce_excessive_speed() -> None:
    points = np.column_stack((np.arange(12) * 2., np.zeros(12), np.ones(12))).astype(np.float32)
    valid: NDArray[np.bool_] = np.ones(12, bool)
    with pytest.raises(ValueError, match="speed limit"):
        smooth_ball_positions(points, valid, fps=60., config=BallSmoothingConfig(method="savgol"))
