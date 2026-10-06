"""The force-model fit explains simulated flights and exposes non-physical ones."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.physics.ball.fitting import FitSettings, fit_flight_physics
from src.utils.physics.ball.record import BallPhysicsRecord
from tests.support.physics.ball_record import simulated_record


def _settings(record_dt: float, stride: int, iterations: int = 30) -> FitSettings:
    return FitSettings(
        gravity=9.8,
        dt=record_dt,
        substeps=stride,
        iterations=iterations,
        minimum_segment_frames=3,
    )


def _spans(record: BallPhysicsRecord) -> list[tuple[int, int]]:
    return [(s.start_frame, s.end_frame) for s in record.segments]


def test_simulated_rallies_are_fitted_exactly_and_share_their_field() -> None:
    first, first_record = simulated_record(48, hits=(50, 132), k_drag=0.012)
    second, second_record = simulated_record(
        40, hits=(70,), k_drag=0.006, wind=(-1.5, 0.8, 0.0)
    )
    fits = fit_flight_physics(
        [first, second],
        [_spans(first_record), _spans(second_record)],
        _settings(first_record.dt, first_record.stride),
        torch.device("cpu"),
    )
    for fit, positions, record in zip(
        fits, (first, second), (first_record, second_record), strict=True
    ):
        assert fit.fitted.all()
        assert fit.residual.shape == positions.shape
        # float32 ground truth: the residual is rounding noise.
        assert np.abs(fit.residual).max() < 1e-4
        assert fit.k_drag == pytest.approx(record.k_drag, rel=1e-2)
        np.testing.assert_allclose(fit.wind_xy, record.wind[:2], atol=0.05)


def test_a_non_physical_wiggle_leaves_a_residual_of_its_size() -> None:
    positions, record = simulated_record(48, hits=(90,))
    frames = np.arange(len(positions))
    wiggled = positions + 0.1 * np.sin(frames / 2.0)[:, None] * np.array([1, 0, 1])
    (fit,) = fit_flight_physics(
        [wiggled],
        [_spans(record)],
        _settings(record.dt, record.stride),
        torch.device("cpu"),
    )
    rmse = float(np.sqrt(np.mean(np.sum(fit.residual**2, axis=-1))))
    assert 0.05 < rmse < 0.15


def test_segments_shorter_than_the_minimum_stay_unfitted() -> None:
    positions, record = simulated_record(40, hits=(150,))  # last segment: 2 frames
    spans = _spans(record)
    assert spans[-1][1] - spans[-1][0] == 2
    (fit,) = fit_flight_physics(
        [positions],
        [spans],
        _settings(record.dt, record.stride, 5),
        torch.device("cpu"),
    )
    assert fit.fitted[: spans[-1][0]].all()
    assert not fit.fitted[spans[-1][0] :].any()


def test_invalid_inputs_are_rejected() -> None:
    positions, record = simulated_record(10)
    with pytest.raises(ValueError, match="one segment list"):
        fit_flight_physics(
            [positions], [], _settings(record.dt, record.stride), torch.device("cpu")
        )
    with pytest.raises(ValueError, match="long enough"):
        fit_flight_physics(
            [positions],
            [[(0, 2)]],
            _settings(record.dt, record.stride),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError):
        FitSettings(
            gravity=9.8, dt=0.01, substeps=1, iterations=1, minimum_segment_frames=2
        )
