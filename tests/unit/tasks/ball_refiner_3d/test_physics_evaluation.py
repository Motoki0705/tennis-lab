"""``physics_eval.v1``: ground truth is physical, a straight-line fill is not."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.data.schema import Rally
from src.tasks.ball_refiner_3d.evaluation.physics import (
    PROTOCOL,
    RallyPrediction,
    physics_metrics,
    physics_report,
)
from tests.support.physics.ball_record import simulated_record


def _rally(
    index: int, frames: int, hits: tuple[int, ...], surface: str, fps: int = 60
) -> Rally:
    positions, record = simulated_record(
        frames, output_fps=fps, sim_fps=240, hits=hits, surface=surface
    )
    return Rally(
        index=index,
        name=f"rally_{index:06d}",
        split="test",
        xyz=positions.astype(np.float32),
        uv=np.zeros((1, frames, 2), np.float32),
        visible=np.ones((1, frames), bool),
        projection=np.zeros((1, 3, 4)),
        events=record.event_mask(),
        time=np.arange(frames) / fps,
        physics=record,
    )


def _prediction(
    rally: Rally, coordinates: np.ndarray, missing: np.ndarray
) -> RallyPrediction:
    probability = np.full(len(rally.xyz), 0.01)
    probability[np.flatnonzero(rally.events)] = 0.9
    return RallyPrediction(rally, coordinates, missing, probability)


@pytest.fixture(scope="module")
def rallies() -> list[Rally]:
    return [
        _rally(0, 90, (120, 260), "clay"),
        _rally(1, 70, (150,), "grass"),
    ]


def test_ground_truth_is_physical_and_events_are_found(rallies: list[Rally]) -> None:
    predictions = [
        _prediction(r, r.xyz.astype(np.float64), np.arange(len(r.xyz)) % 3 == 0)
        for r in rallies
    ]
    report = physics_report(predictions, torch.device("cpu"))
    assert report["protocol"] == PROTOCOL
    assert report["fit_residual_rmse"]["all"] < 1e-4
    assert report["fit_residual_rmse"]["missing"] < 1e-4
    assert report["kinematics"]["all"]["acceleration_rmse"] == 0.0
    assert report["kinematics"]["all"]["implausible_acceleration_rate"] == 0.0
    assert report["events"]["recall"] == 1.0 and report["events"]["precision"] == 1.0
    assert set(report["breakdown"]["surface"]) == {"clay", "grass"}
    metrics = physics_metrics(report)
    assert metrics["test_event_f1"] == 1.0
    assert metrics["test_jerk_ratio"] == pytest.approx(1.0)


def test_linear_gap_filling_is_flagged_inside_the_gap(rallies: list[Rally]) -> None:
    predictions = []
    # Gaps inside one flight segment: rally 0 [30,65), rally 1 [0,38).
    for rally, (start, end) in zip(rallies, ((35, 60), (5, 33)), strict=True):
        assert len({s for s in rally.physics.frame_segment()[start - 1 : end + 1]}) == 1
        coordinates = rally.xyz.astype(np.float64)
        missing: NDArray[np.bool_] = np.zeros(len(coordinates), bool)
        missing[start:end] = True
        weight = np.linspace(0, 1, end - start + 2)[1:-1, None]
        coordinates[start:end] = (1 - weight) * coordinates[start - 1] + (
            weight * coordinates[end]
        )
        predictions.append(_prediction(rally, coordinates, missing))
    report = physics_report(predictions, torch.device("cpu"), events=False)
    residual = report["fit_residual_rmse"]
    # One consistent flight cannot follow the chord.  The fit spreads the
    # mismatch over the whole segment, so observed frames carry it as well.
    assert residual["missing"] > 0.02
    assert residual["observed"] > 0.01
    # A straight line has no gravity: in-gap acceleration error is about g.
    assert report["kinematics"]["missing"]["acceleration_rmse"] > 5.0
    assert "events" not in report
    with pytest.raises(KeyError):
        physics_metrics(report)


def test_rallies_must_share_one_output_rate(rallies: list[Rally]) -> None:
    other = _rally(2, 40, (), "hard", fps=30)
    predictions = [
        _prediction(r, r.xyz.astype(np.float64), np.zeros(len(r.xyz), bool))
        for r in (rallies[0], other)
    ]
    with pytest.raises(ValueError, match="output FPS"):
        physics_report(predictions, torch.device("cpu"))
