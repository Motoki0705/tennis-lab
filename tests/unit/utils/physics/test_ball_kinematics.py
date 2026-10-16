"""In-flight finite differences never straddle an impact."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.physics.ball.kinematics import (
    KinematicThresholds,
    kinematic_report,
    segment_difference,
    stencil_any,
)

FPS = 60.0
ACCELERATION = np.array([0.5, -1.0, -9.8])


def _parabola(frames: int, start: float = 0.0) -> NDArray[np.float64]:
    t = start + np.arange(frames)[:, None] / FPS
    result: NDArray[np.float64] = (
        np.array([0.0, -10.0, 1.0])
        + np.array([2.0, 20.0, 5.0]) * t
        + (0.5 * ACCELERATION * t**2)
    )
    return result


def test_second_difference_of_a_parabola_is_its_acceleration() -> None:
    positions = _parabola(12)
    values, valid = segment_difference(positions, np.zeros(12, np.int64), FPS, 2)
    assert valid[:10].all() and not valid[10:].any()
    np.testing.assert_allclose(values[:10], np.tile(ACCELERATION, (10, 1)), atol=1e-8)
    jerk, _ = segment_difference(positions, np.zeros(12, np.int64), FPS, 3)
    np.testing.assert_allclose(jerk[:9], 0.0, atol=1e-6)


def test_stencils_crossing_a_boundary_or_unlabelled_frames_are_invalid() -> None:
    segment = np.array([0, 0, 0, 0, 1, 1, 1, -1, 2, 2, 2])
    _, valid = segment_difference(np.zeros((11, 3)), segment, FPS, 2)
    # Anchors k use frames k..k+2.
    assert valid.tolist() == [
        True, True, False, False, True, False, False, False, True, False, False,
    ]  # fmt: skip


def test_stencil_any_extends_a_mask_backwards_over_the_stencil() -> None:
    mask: NDArray[np.bool_] = np.zeros(8, bool)
    mask[5] = True
    assert np.flatnonzero(stencil_any(mask, 2)).tolist() == [3, 4, 5]


def test_report_is_zero_for_ground_truth_and_flags_impossible_motion() -> None:
    frames = 30
    target = np.concatenate((_parabola(15), _parabola(15, start=0.1)))
    segment: NDArray[np.int64] = np.repeat([0, 1], 15)
    groups = {"all": np.ones(frames, bool), "second": segment == 1}
    thresholds = KinematicThresholds(acceleration_mps2=40.0, below_ground_m=0.02)

    exact = kinematic_report(target, target, segment, FPS, groups, thresholds)
    assert exact["all"]["velocity_rmse"] == 0.0
    assert exact["all"]["acceleration_rmse"] == 0.0
    assert exact["all"]["jerk_ratio"] == pytest.approx(1.0)
    assert exact["all"]["implausible_acceleration_rate"] == 0.0
    # 13 valid second differences per 15-frame segment; none crosses frame 15.
    assert exact["all"]["acceleration_count"] == 26
    assert exact["second"]["acceleration_count"] == 13

    spiked = target.copy()
    spiked[20, 2] -= 0.5  # half a metre at 60 fps: ~1800 m/s^2, under the court
    report = kinematic_report(spiked, target, segment, FPS, groups, thresholds)
    # Frame 20 enters the stencils anchored at 18, 19 and 20.
    assert report["second"]["implausible_acceleration_rate"] == pytest.approx(3 / 13)
    assert report["second"]["below_ground_rate"] == 0.0
    assert report["all"]["acceleration_rmse"] > 100
    assert report["all"]["jerk_ratio"] > 10

    sunk = target - np.array([0.0, 0.0, 10.0])
    below = kinematic_report(sunk, target, segment, FPS, groups, thresholds)
    assert below["all"]["below_ground_rate"] == 1.0
    assert below["all"]["acceleration_rmse"] == pytest.approx(0.0, abs=1e-8)


def test_empty_groups_report_undefined_values_not_zero() -> None:
    target = _parabola(6)
    report = kinematic_report(
        target,
        target,
        np.zeros(6, np.int64),
        FPS,
        {"none": np.zeros(6, bool)},
        KinematicThresholds(40.0, 0.02),
    )["none"]
    assert report["velocity_rmse"] is None
    assert report["velocity_count"] == 0
    assert report["implausible_acceleration_rate"] is None
    assert report["below_ground_rate"] is None
