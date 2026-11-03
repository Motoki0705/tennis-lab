"""Constant-velocity Kalman/RTS estimates, with independent contiguous segments."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.probabilistic_triangulation.distributions import (
    validate_covariance,
)


def smooth_positions(
    positions: NDArray[np.float64], covariance: NDArray[np.float64],
    valid: NDArray[np.bool_], *, fps: float, acceleration_std: float,
    initial_velocity_std: float, backward: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """White-acceleration CV model; gaps reset state and stay zero in the output.

    Position units are arbitrary, velocity is units/second, and acceleration_std
    is units/second². Each interval has an independent constant acceleration.
    Missing measurements are not predicted/exported. RTS never crosses a gap.
    """
    t = len(positions)
    if (positions.shape != (t, 3) or covariance.shape != (t, 3, 3)
            or valid.shape != (t,) or valid.dtype != bool
            or not np.isfinite(positions).all() or not np.isfinite(covariance).all()):
        raise ValueError("Kalman position/covariance/mask axes must agree and be finite")
    if any(not math.isfinite(x) or x <= 0 for x in (fps, acceleration_std, initial_velocity_std)):
        raise ValueError("Kalman FPS and noise scales must be positive finite")
    if type(backward) is not bool:
        raise ValueError("Kalman backward must be bool")
    validate_covariance(covariance[valid], (int(valid.sum()), 3, 3))
    dt = 1 / fps
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    drive = np.concatenate((np.eye(3) * dt**2 / 2, np.eye(3) * dt))
    process = drive @ drive.T * acceleration_std**2
    means = np.zeros((t, 6))
    covs = np.zeros((t, 6, 6))
    predicted = np.zeros_like(means)
    predicted_covs = np.zeros_like(covs)
    eye = np.eye(6)
    for i in range(t):
        if not valid[i]:
            continue
        if i == 0 or not valid[i - 1]:
            means[i, :3] = positions[i]
            covs[i, :3, :3] = covariance[i]
            covs[i, 3:, 3:] = np.eye(3) * initial_velocity_std**2
            continue
        predicted[i] = transition @ means[i - 1]
        predicted_covs[i] = transition @ covs[i - 1] @ transition.T + process
        innovation_cov = predicted_covs[i, :3, :3] + covariance[i]
        gain = np.linalg.solve(innovation_cov, predicted_covs[i, :3, :]).T
        means[i] = predicted[i] + gain @ (positions[i] - predicted[i, :3])
        residual = eye.copy()
        residual[:, :3] -= gain
        covs[i] = residual @ predicted_covs[i] @ residual.T + gain @ covariance[i] @ gain.T
        covs[i] = (covs[i] + covs[i].T) / 2
    if backward:
        for i in range(t - 2, -1, -1):
            if not valid[i] or not valid[i + 1]:
                continue
            gain = np.linalg.solve(predicted_covs[i + 1], transition @ covs[i]).T
            means[i] += gain @ (means[i + 1] - predicted[i + 1])
            covs[i] += gain @ (covs[i + 1] - predicted_covs[i + 1]) @ gain.T
            covs[i] = (covs[i] + covs[i].T) / 2
    return means[:, :3].copy(), covs[:, :3, :3].copy()
