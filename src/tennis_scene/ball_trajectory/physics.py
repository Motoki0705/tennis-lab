"""Batched RK4 integration of gravity and quadratic aerodynamic drag."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def flight_states(
    initial: NDArray[np.float64],
    times: NDArray[np.float64],
    drag_per_m: float,
    *,
    gravity_mps2: float,
    steps: int,
) -> NDArray[np.float64]:
    """One six-component initial state per time, including small negative times.

    Independent trajectories are integrated together without allocating a dense
    time/heatmap volume. Times are seconds relative to each segment's origin.
    """
    state = np.array(initial, dtype=np.float64, copy=True)
    if (
        state.shape != (len(times), 6)
        or not np.isfinite(state).all()
        or not np.isfinite(times).all()
    ):
        raise ValueError("Flight state/time axes must be finite (N,6)/(N)")
    if (
        not np.isfinite(drag_per_m)
        or drag_per_m < 0
        or not np.isfinite(gravity_mps2)
        or gravity_mps2 <= 0
        or type(steps) is not int
        or steps < 1
    ):
        raise ValueError("Invalid physical integration settings")
    dt = times[:, None] / steps

    def derivative(y: NDArray[np.float64]) -> NDArray[np.float64]:
        velocity = y[:, 3:]
        acceleration = (
            -drag_per_m * np.linalg.norm(velocity, axis=1)[:, None] * velocity
        )
        acceleration[:, 2] -= gravity_mps2
        return np.concatenate((velocity, acceleration), axis=1)

    for _ in range(steps):
        k1 = derivative(state)
        k2 = derivative(state + dt * k1 / 2)
        k3 = derivative(state + dt * k2 / 2)
        k4 = derivative(state + dt * k3)
        state += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state
