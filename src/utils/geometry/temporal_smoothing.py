"""Deterministic local motion consensus and robust quadratic LOESS for 2D points.

Observed, rejected and interpolated points have separate masks. A short sequence
without enough local support retains its measurements with ``smoothed=False``;
neither extrapolation nor interpolation creates an observed measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TemporalSmoothingConfig:
    enabled: bool = True
    consensus_seconds: float = 0.7
    trials: int = 80
    minimum_residual_diagonal: float = 0.012
    residual_mad_multiplier: float = 4.0
    max_gap_seconds: float = 0.20
    loess_seconds: float = 0.35
    seed: int = 42

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("Smoothing enabled must be bool")
        for name in ("consensus_seconds", "minimum_residual_diagonal", "residual_mad_multiplier", "loess_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_gap_seconds) not in (int, float) or not math.isfinite(self.max_gap_seconds) or self.max_gap_seconds < 0:
            raise ValueError("max_gap_seconds must be finite and nonnegative")
        if type(self.trials) is not int or self.trials < 1:
            raise ValueError("trials must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class TemporalSmoothingResult:
    coordinates: NDArray[np.float32]
    source_observed: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    rejected: NDArray[np.bool_]
    interpolated: NDArray[np.bool_]
    smoothed: NDArray[np.bool_]
    consensus_supported: NDArray[np.bool_]
    restored_turn: NDArray[np.bool_]

    def __post_init__(self) -> None:
        n = len(self.source_observed)
        masks = (self.source_observed, self.accepted, self.rejected, self.interpolated,
                 self.smoothed, self.consensus_supported, self.restored_turn)
        if any(mask.shape != (n,) or mask.dtype != np.bool_ for mask in masks):
            raise ValueError("Temporal masks must be bool arrays on one frame axis")
        if self.coordinates.shape != (n, 2) or self.coordinates.dtype != np.float32 or not np.isfinite(self.coordinates).all():
            raise ValueError("Temporal coordinates must be finite float32 (T, 2)")
        if np.any(self.accepted & self.rejected) or not np.array_equal(self.accepted | self.rejected, self.source_observed):
            raise ValueError("Accepted and rejected must partition source observations")
        if np.any(self.accepted & self.interpolated) or np.any(self.smoothed & ~(self.accepted | self.interpolated)):
            raise ValueError("Interpolation must stay distinct from accepted observations")
        if np.any(self.consensus_supported & ~self.source_observed) or np.any(self.restored_turn & ~self.accepted):
            raise ValueError("Consensus and restored turns must be supported by observations")
        if np.any(self.coordinates[~(self.accepted | self.interpolated)] != 0):
            raise ValueError("Missing temporal coordinates must be zero")


DEFAULT_TEMPORAL_SMOOTHING = TemporalSmoothingConfig()


def _segments(ids: NDArray[np.int64], max_gap: int) -> list[NDArray[np.int64]]:
    if not len(ids):
        return []
    return list(np.split(ids, np.flatnonzero(np.diff(ids) - 1 > max_gap) + 1))


def _consensus(
    points: NDArray[np.float64], ids: NDArray[np.int64], i: int, half: int,
    config: TemporalSmoothingConfig, rng: np.random.Generator,
) -> tuple[NDArray[np.float64], float] | None:
    if len(ids) < 9:
        return None
    dt = (ids - i) / half
    basis = np.column_stack((np.ones(len(ids)), dt, dt * dt))
    values = points[ids]
    best, score = None, float("inf")
    # Include a well-spaced deterministic candidate; random trials use a pinned seed.
    candidates = [np.array([0, len(ids) // 2, len(ids) - 1])]
    candidates.extend(np.sort(rng.choice(len(ids), 3, replace=False)) for _ in range(config.trials))
    for choice in candidates:
        if np.min(np.diff(dt[choice])) < 0.15:
            continue
        coef = np.linalg.solve(basis[choice], values[choice])
        residual = np.linalg.norm(basis @ coef - values, axis=1)
        candidate_score = float(np.median(residual))
        if candidate_score < score:
            best, score = coef, candidate_score
    if best is None:
        return None
    threshold = config.minimum_residual_diagonal
    for _ in range(4):
        residual = np.linalg.norm(basis @ best - values, axis=1)
        scale = max(config.minimum_residual_diagonal / 3, 1.4826 * float(np.median(abs(residual - np.median(residual)))))
        threshold = max(config.minimum_residual_diagonal, config.residual_mad_multiplier * scale)
        support = residual <= threshold
        if support.sum() < 6:
            break
        best = np.linalg.lstsq(basis[support], values[support], rcond=None)[0]
    return best[0], threshold


def _restore_turns(
    points: NDArray[np.float64], observed: NDArray[np.bool_], accepted: NDArray[np.bool_], fps: float,
) -> NDArray[np.bool_]:
    restored: NDArray[np.bool_] = np.zeros(len(points), bool)
    context_frames = max(2, round(5 * fps / 60))
    for group in _segments(np.flatnonzero(observed & ~accepted), 0):
        if len(group) > max(1, round(.067 * fps)):
            continue
        left, right = int(group[0]) - 1, int(group[-1]) + 1
        if left < 0 or right >= len(points) or not accepted[left] or not accepted[right] or not observed[left:right + 1].all():
            continue
        speeds = []
        for start, stop in ((max(0, left - context_frames), left + 1), (right, min(len(points), right + context_frames + 1))):
            for k in range(start + 1, stop):
                if accepted[k - 1] and accepted[k]:
                    speeds.append(float(np.linalg.norm(points[k] - points[k - 1])))
        if len(speeds) < 4:
            continue
        limit = max(.018 * 60 / fps, 3 * float(np.median(speeds)))
        if np.linalg.norm(np.diff(points[left:right + 1], axis=0), axis=1).max() <= limit:
            accepted[group] = True
            restored[group] = True
    return restored


def smooth_2d_trajectory(
    xy: NDArray[np.floating], confidence: NDArray[np.floating], observed: NDArray[np.bool_], *,
    fps: float, image_size_wh: tuple[int, int], config: TemporalSmoothingConfig = DEFAULT_TEMPORAL_SMOOTHING,
) -> TemporalSmoothingResult:
    """Reject transient excursions and smooth retained measurements, without manual intervals.

    Distances use the image diagonal; temporal windows use source FPS. Long
    gaps separate local fits. Only bounded gaps up to ``max_gap_seconds`` can
    have interpolated coordinates; callers must use ``accepted`` for geometry.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    if len(image_size_wh) != 2 or any(type(v) is not int or v < 2 for v in image_size_wh):
        raise ValueError("Image size must contain two integer dimensions >= 2")
    points = np.asarray(xy, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    n = len(points)
    if points.shape != (n, 2) or scores.shape != (n,) or observed.shape != (n,) or observed.dtype != np.bool_:
        raise ValueError("Expected coordinates (T,2), confidence (T,) and bool observed (T,)")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)) or not np.isfinite(points[observed]).all():
        raise ValueError("Observed coordinates and confidence must be finite; confidence must be in [0,1]")
    accepted = observed.copy()
    rejected: NDArray[np.bool_] = np.zeros(n, bool)
    interpolated: NDArray[np.bool_] = np.zeros(n, bool)
    smoothed: NDArray[np.bool_] = np.zeros(n, bool)
    supported: NDArray[np.bool_] = np.zeros(n, bool)
    restored: NDArray[np.bool_] = np.zeros(n, bool)
    output = np.where(observed[:, None], points, 0).astype(np.float32)
    if not config.enabled or not observed.any():
        return TemporalSmoothingResult(output, observed.copy(), accepted, rejected, interpolated, smoothed, supported, restored)
    diagonal = math.hypot(*image_size_wh)
    normalized = points / diagonal
    half = max(6, round(config.consensus_seconds * fps / 2))
    max_gap = round(config.max_gap_seconds * fps)
    rng = np.random.default_rng(config.seed)
    for segment in _segments(np.flatnonzero(observed), max_gap):
        for i in segment:
            ids = segment[abs(segment - i) <= half]
            fit = _consensus(normalized, ids, int(i), half, config, rng)
            if fit is not None:
                prediction, threshold = fit
                supported[i] = True
                accepted[i] = np.linalg.norm(normalized[i] - prediction) <= threshold
    restored = _restore_turns(normalized, observed, accepted, fps)
    rejected = observed & ~accepted
    output[~accepted] = 0
    half_loess = max(3, round(config.loess_seconds * fps / 2))
    for segment in _segments(np.flatnonzero(accepted), max_gap):
        for i in range(int(segment[0]), int(segment[-1]) + 1):
            near = segment[abs(segment - i) <= half_loess]
            if len(near) < 4:
                near = segment[np.argsort(abs(segment - i), kind="stable")[:min(8, len(segment))]]
            if len(near) < 3:
                continue  # Accepted raw measurements remain, with smoothed=False.
            d = (near - i).astype(np.float64)
            radius = max(float(abs(d).max()) * 1.0001, 1.)
            x = d / radius
            basis = np.column_stack((np.ones(len(x)), x, x * x))
            base = (1 - abs(x) ** 3) ** 3 * np.maximum(scores[near], .1) ** 1.2
            weights = base.copy()
            for _ in range(4):
                root = np.sqrt(np.maximum(weights, 1e-8))
                coef = np.linalg.lstsq(basis * root[:, None], points[near] * root[:, None], rcond=None)[0]
                residual = np.linalg.norm(basis @ coef - points[near], axis=1)
                scale = max(diagonal * .001, 1.4826 * float(np.median(residual)))
                weights = base * np.minimum(1, 1.4 * scale / np.maximum(residual, 1e-8))
            output[i] = coef[0]
            smoothed[i] = True
            interpolated[i] = not accepted[i]
    return TemporalSmoothingResult(output, observed.copy(), accepted, rejected, interpolated, smoothed, supported, restored)
