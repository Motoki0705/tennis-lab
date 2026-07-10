"""Kinematics extracted from a 3D ball trajectory: per-frame speed and bounces.

Pure-NumPy helpers (no matplotlib/torch dependency) shared by scene rendering
(HUD speed readout, bounce markers) and any evaluation code that needs
physical quantities derived from a ``(T, 3)`` court-coordinate ball track.

Missing observations are represented as non-finite rows (NaN/inf); every
function documents explicitly how such frames propagate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_positions(positions: NDArray[np.float32]) -> NDArray[np.float64]:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must have shape (T, 3), got {arr.shape}")
    return arr


def compute_speeds(
    positions: NDArray[np.float32],
    fps: float,
) -> NDArray[np.float32]:
    """Per-frame speed magnitude (m/s) of a 3D trajectory.

    Uses central differences on the interior and one-sided differences at the
    boundaries. A frame's speed falls back to a one-sided difference when one
    neighbour is missing, and is NaN when no finite difference can be formed.

    Args:
        positions: Trajectory of shape (T, 3); missing frames are non-finite.
        fps: Frames per second of the trajectory. Must be positive.

    Returns:
        Array of shape (T,) with speeds in m/s; NaN where not computable.
    """
    arr = _validate_positions(positions)
    if fps <= 0.0 or not np.isfinite(fps):
        raise ValueError(f"fps must be a positive finite number, got {fps}")

    num_frames = arr.shape[0]
    speeds = np.full(num_frames, np.nan, dtype=np.float64)
    if num_frames < 2:
        return speeds.astype(np.float32)

    valid = np.isfinite(arr).all(axis=1)
    for t in range(num_frames):
        prev_ok = t - 1 >= 0 and valid[t - 1]
        next_ok = t + 1 < num_frames and valid[t + 1]
        if prev_ok and next_ok:
            delta = arr[t + 1] - arr[t - 1]
            dt = 2.0 / fps
        elif valid[t] and next_ok:
            delta = arr[t + 1] - arr[t]
            dt = 1.0 / fps
        elif valid[t] and prev_ok:
            delta = arr[t] - arr[t - 1]
            dt = 1.0 / fps
        else:
            continue
        speeds[t] = float(np.linalg.norm(delta)) / dt

    return speeds.astype(np.float32)


def detect_bounces(
    positions: NDArray[np.float32],
    *,
    max_bounce_height: float = 0.4,
    min_prominence: float = 0.02,
    min_separation: int = 5,
) -> NDArray[np.int64]:
    """Detect ground-bounce frames as prominent local minima of the Z track.

    A frame ``t`` is a bounce candidate when ``z[t]`` and both neighbours are
    finite, ``z[t]`` is a local minimum (``z[t] <= z[t-1]`` and
    ``z[t] < z[t+1]``), the ball is near the ground
    (``z[t] < max_bounce_height``), and the V-shape around it is prominent
    (``(z[t-1] - z[t]) + (z[t+1] - z[t]) >= min_prominence``). Candidates
    closer than ``min_separation`` frames are merged, keeping the lowest one.

    Args:
        positions: Trajectory of shape (T, 3); missing frames are non-finite.
        max_bounce_height: Maximum Z (metres) for a minimum to count as a
            ground bounce rather than a racket hit.
        min_prominence: Minimum combined descent+ascent depth (metres) around
            the minimum, filtering flat noise.
        min_separation: Minimum frame gap between two reported bounces.

    Returns:
        Sorted array of bounce frame indices (possibly empty).
    """
    arr = _validate_positions(positions)
    if max_bounce_height <= 0.0:
        raise ValueError(f"max_bounce_height must be positive, got {max_bounce_height}")
    if min_prominence < 0.0:
        raise ValueError(f"min_prominence must be non-negative, got {min_prominence}")
    if min_separation < 1:
        raise ValueError(f"min_separation must be >= 1, got {min_separation}")

    z = arr[:, 2]
    valid = np.isfinite(arr).all(axis=1)

    candidates: list[int] = []
    for t in range(1, arr.shape[0] - 1):
        if not (valid[t - 1] and valid[t] and valid[t + 1]):
            continue
        if z[t] >= max_bounce_height:
            continue
        if not (z[t] <= z[t - 1] and z[t] < z[t + 1]):
            continue
        if (z[t - 1] - z[t]) + (z[t + 1] - z[t]) < min_prominence:
            continue
        candidates.append(t)

    merged: list[int] = []
    for t in candidates:
        if merged and t - merged[-1] < min_separation:
            if z[t] < z[merged[-1]]:
                merged[-1] = t
        else:
            merged.append(t)

    return np.asarray(merged, dtype=np.int64)
