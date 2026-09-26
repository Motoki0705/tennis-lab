"""Event-aware temporal smoothing of an already triangulated single ball."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationOutput
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints

BallSmoothingMethod = Literal["none", "savgol", "robust_spline", "ballistic_rts"]


@dataclass(frozen=True)
class BallSmoothingConfig:
    method: BallSmoothingMethod = "none"
    window_frames: int = 11
    position_sigma_m: float = 0.07
    spline_acceleration_sigma_mps2: float = 25.0
    rts_acceleration_sigma_mps2: float = 15.0
    huber_delta_m: float = 0.12
    event_y_prominence_m: float = 1.0
    event_bounce_height_m: float = 0.45
    event_z_prominence_m: float = 0.3
    event_separation_frames: int = 8

    def __post_init__(self) -> None:
        if self.method not in ("none", "savgol", "robust_spline", "ballistic_rts"):
            raise ValueError(f"Unknown ball smoothing method: {self.method}")
        if self.window_frames < 5 or self.window_frames % 2 != 1:
            raise ValueError("Ball smoothing window_frames must be odd and at least five")
        values = (self.position_sigma_m, self.spline_acceleration_sigma_mps2,
                  self.rts_acceleration_sigma_mps2, self.huber_delta_m,
                  self.event_y_prominence_m, self.event_bounce_height_m,
                  self.event_z_prominence_m)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Ball smoothing scales and event thresholds must be finite and positive")
        if self.event_separation_frames < 1:
            raise ValueError("Ball event separation must be positive")


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


def _savgol(values: NDArray[np.float64], config: BallSmoothingConfig) -> NDArray[np.float64]:
    window = min(config.window_frames, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return values.copy()
    return np.asarray(savgol_filter(values, window, 2, axis=0, mode="interp"), np.float64)


def _robust_spline(values: NDArray[np.float64], fps: float, config: BallSmoothingConfig) -> NDArray[np.float64]:
    count = len(values)
    if count < 4:
        return values.copy()
    second = diags((np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)), (0, 1, 2), shape=(count - 2, count), format="csc")
    strength = (config.position_sigma_m * fps * fps / config.spline_acceleration_sigma_mps2) ** 2
    curvature = strength * (second.T @ second)
    weights: NDArray[np.float64] = np.ones(count, np.float64)
    weights[[0, -1]] = 5.0
    result = values.copy()
    for _ in range(5):
        system = diags(weights, format="csc") + curvature
        result = np.column_stack([spsolve(system, weights * values[:, axis]) for axis in range(3)])
        residual = np.linalg.norm(result - values, axis=1)
        weights = np.minimum(1.0, config.huber_delta_m / np.maximum(residual, 1e-12))
        weights[[0, -1]] = 5.0
    return np.asarray(result, np.float64)


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


def smooth_ball_positions(
    positions: NDArray[np.float32], valid: NDArray[np.bool_], *, fps: float, config: BallSmoothingConfig,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Keep original support and event positions; smooth only within supported flights."""
    if positions.shape != (len(valid), 3) or valid.dtype != np.bool_ or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Ball smoothing requires (T,3) positions, a boolean mask, and positive FPS")
    if not np.isfinite(positions).all() or (positions[~valid] != 0).any():
        raise ValueError("Ball positions must be finite and zero outside the validity mask")
    if config.method == "none":
        return positions.copy(), np.zeros(0, np.int32)
    events = ball_event_frames(positions, valid, config)
    result = np.asarray(positions, np.float64).copy()
    for start, end in _segments(valid, events):
        segment = np.asarray(positions[start:end], np.float64)
        if config.method == "savgol":
            result[start:end] = _savgol(segment, config)
        elif config.method == "robust_spline":
            result[start:end] = _robust_spline(segment, fps, config)
        else:
            result[start:end] = _ballistic_rts(segment, fps, config)
    result[events] = positions[events]
    if not np.isfinite(result).all() or (result[valid, 2] < -0.2).any() or (result[valid, 2] > 20).any():
        raise ValueError("Smoothed ball trajectory violates geometric bounds")
    result[~valid] = 0
    return result.astype(np.float32), events


def _reprojection_errors(
    positions: NDArray[np.float32], valid: NDArray[np.bool_], uv_px: NDArray[np.float32],
    visibility: NDArray[np.bool_], cameras: tuple[PinholeCamera, ...],
) -> NDArray[np.float32]:
    if uv_px.shape != (len(cameras), len(valid), 2) or visibility.shape != (len(cameras), len(valid)):
        raise ValueError("Ball UV and camera axes disagree with the triangulated timeline")
    errors = np.zeros(visibility.shape, np.float32)
    for view, camera in enumerate(cameras):
        projected, front = camera.project(positions)
        supported = valid & visibility[view] & front
        errors[view, supported] = np.linalg.norm(projected[supported] - uv_px[view, supported], axis=-1).astype(np.float32)
    return errors


@dataclass(frozen=True)
class BallSmoothingInput:
    source: ClipSource
    triangulation: BallTriangulationOutput
    alignment: CameraAlignmentOutput


class BallSmoothingModule:
    def __init__(self, config: BallSmoothingConfig) -> None:
        self.config = config
        self.io = ComponentIO(
            "ball_smoothing", BallSmoothingInput, BallTriangulationOutput,
            {"triangulation": InputPort("ball_trajectory"), "alignment": InputPort("aligned_cameras")},
            "smoothed_ball_trajectory",
        )

    def process(self, inputs: BallSmoothingInput) -> BallTriangulationOutput:
        raw = inputs.triangulation.ball
        if raw is None or self.config.method == "none":
            return inputs.triangulation
        geometry = inputs.alignment.geometry
        if geometry is None:
            raise ValueError("Ball smoothing requires the alignment used for triangulation")
        trajectory = raw.trajectory
        positions, _ = smooth_ball_positions(trajectory.positions, trajectory.valid, fps=inputs.source.fps, config=self.config)
        smoothed = TriangulatedPoints(positions, trajectory.valid.copy(), trajectory.reasons.copy(),
            trajectory.inliers.copy(), _reprojection_errors(positions, trajectory.valid, raw.uv_px, raw.visibility, geometry.cameras))
        return BallTriangulationOutput(BallReconstructionResult(raw.uv_px, raw.visibility, smoothed, raw.status))
