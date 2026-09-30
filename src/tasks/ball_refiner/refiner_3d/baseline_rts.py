"""Frozen #929 event-aware gravity RTS core for a synthetic diagnostic baseline.

Source: 0f124818ea97dbec716e20a7fa6da315edb8ee82,
src/tennis_scene/pipeline/components/ball_smoothing.py.
The Kalman/RTS passes and event segmentation below retain their original math.
The synthetic adapter reports pipeline bounds violations instead of rejecting
rallies; it never clips, drops frames, or uses ground-truth events.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


@dataclass(frozen=True)
class BallSmoothingConfig:
    position_sigma_m: float = 0.07
    rts_acceleration_sigma_mps2: float = 15.0
    huber_delta_m: float = 0.12
    event_y_prominence_m: float = 1.0
    event_bounce_height_m: float = 0.45
    event_z_prominence_m: float = 0.3
    event_separation_frames: int = 8


def _valid_runs(valid: NDArray[np.bool_]) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist(), strict=True))


def ball_event_frames(positions: NDArray[np.float32], valid: NDArray[np.bool_], config: BallSmoothingConfig) -> NDArray[np.int32]:
    """Detect bounce minima and stroke-direction reversals without bridging missing frames."""
    if positions.shape != (len(valid), 3) or valid.dtype != np.bool_ or not np.isfinite(positions[valid]).all():
        raise ValueError("Ball events require finite (T,3) positions and a boolean (T,) mask")
    events: list[int] = []
    for start, end in _valid_runs(valid):
        if end - start < 9:
            continue
        pilot = median_filter(np.asarray(positions[start:end], np.float64), size=(5, 1), mode="nearest")
        y, z = pilot[:, 1], pilot[:, 2]
        maxima, _ = find_peaks(y, prominence=config.event_y_prominence_m, distance=config.event_separation_frames)
        minima, _ = find_peaks(-y, prominence=config.event_y_prominence_m, distance=config.event_separation_frames)
        bounces, _ = find_peaks(-z, prominence=config.event_z_prominence_m, distance=config.event_separation_frames)
        bounces = bounces[z[bounces] <= config.event_bounce_height_m]
        candidates = sorted(set(np.r_[maxima, minima, bounces].tolist()))
        for local in candidates:
            frame = start + local
            if frame <= start or frame >= end - 1:
                continue
            if events and frame - events[-1] < config.event_separation_frames and events[-1] >= start:
                if positions[frame, 2] < positions[events[-1], 2]:
                    events[-1] = frame
            else:
                events.append(frame)
    return np.asarray(events, np.int32)


def _segments(valid: NDArray[np.bool_], events: NDArray[np.int32]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for start, end in _valid_runs(valid):
        cuts = [start, *(int(event) for event in events if start < event < end - 1), end - 1]
        segments.extend((left, right + 1) for left, right in zip(cuts[:-1], cuts[1:], strict=True))
    return segments


def _rts_pass(values: NDArray[np.float64], fps: float, config: BallSmoothingConfig, weights: NDArray[np.float64]) -> NDArray[np.float64]:
    count, dt = len(values), 1.0 / fps
    identity = np.eye(3, dtype=np.float64)
    transition = np.eye(6, dtype=np.float64)
    transition[:3, 3:] = dt * identity
    gravity: NDArray[np.float64] = np.zeros(6, np.float64)
    gravity[2], gravity[5] = -0.5 * 9.81 * dt * dt, -9.81 * dt
    noise_map = np.vstack((0.5 * dt * dt * identity, dt * identity))
    process = (config.rts_acceleration_sigma_mps2 ** 2) * (noise_map @ noise_map.T)
    observe = np.hstack((identity, np.zeros((3, 3), np.float64)))
    measurement = (config.position_sigma_m ** 2) * identity
    filtered: NDArray[np.float64] = np.zeros((count, 6), np.float64)
    filtered_cov: NDArray[np.float64] = np.zeros((count, 6, 6), np.float64)
    predicted = np.zeros_like(filtered)
    predicted_cov = np.zeros_like(filtered_cov)
    velocity = np.median(np.diff(values[:min(count, 6)], axis=0), axis=0) * fps
    filtered[0, :3], filtered[0, 3:] = values[0], velocity
    filtered_cov[0] = np.diag([config.position_sigma_m ** 2] * 3 + [100.0] * 3)
    for frame in range(1, count):
        predicted[frame] = transition @ filtered[frame - 1] + gravity
        predicted_cov[frame] = transition @ filtered_cov[frame - 1] @ transition.T + process
        innovation_cov = observe @ predicted_cov[frame] @ observe.T + measurement / weights[frame]
        gain = np.linalg.solve(innovation_cov, observe @ predicted_cov[frame]).T
        filtered[frame] = predicted[frame] + gain @ (values[frame] - observe @ predicted[frame])
        joseph = np.eye(6) - gain @ observe
        filtered_cov[frame] = joseph @ predicted_cov[frame] @ joseph.T + gain @ (measurement / weights[frame]) @ gain.T
    smoothed = filtered.copy()
    for frame in range(count - 2, -1, -1):
        gain = np.linalg.solve(predicted_cov[frame + 1], transition @ filtered_cov[frame]).T
        smoothed[frame] += gain @ (smoothed[frame + 1] - predicted[frame + 1])
    return smoothed[:, :3]


def _ballistic_rts(values: NDArray[np.float64], fps: float, config: BallSmoothingConfig) -> NDArray[np.float64]:
    if len(values) < 3:
        return values.copy()
    weights: NDArray[np.float64] = np.ones(len(values), np.float64)
    weights[[0, -1]] = 4.0
    result = _rts_pass(values, fps, config, weights)
    for _ in range(2):
        residual = np.linalg.norm(result - values, axis=1)
        weights = np.minimum(1.0, config.huber_delta_m / np.maximum(residual, 1e-12))
        weights[[0, -1]] = 4.0
        result = _rts_pass(values, fps, config, weights)
    return result


def smooth_mixture_mean(positions: NDArray[Any], *, fps: float) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Apply the unretuned #929 core to every finite mixture-mean frame."""
    points = np.asarray(positions, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all() or not math.isfinite(fps) or fps <= 0:
        raise ValueError('RTS baseline requires finite T,3 positions and positive FPS')
    config = BallSmoothingConfig()
    valid: NDArray[np.bool_] = np.ones(len(points), dtype=bool)
    events = ball_event_frames(points, valid, config)
    result = points.astype(np.float64)
    for start, end in _segments(valid, events):
        result[start:end] = _ballistic_rts(points[start:end].astype(np.float64), fps, config)
    result[events] = points[events]
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite RTS result; no fallback')
    bounds = (np.abs(result[:, :2]) > 40).any(-1) | (result[:, 2] < -.2) | (result[:, 2] > 20)
    speed = np.linalg.norm(np.diff(result, axis=0), axis=-1) * fps
    return result.astype(np.float32), {'event_frames': events.tolist(),
        'geometric_bounds_violations': int(bounds.sum()), 'speed_above_65_mps': int((speed > 65).sum()),
        'support': 'all finite condition frames including gaps; pipeline bounds are diagnostics, no rejection'}
