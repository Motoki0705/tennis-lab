"""One event/noise realization feeds both coordinate refiners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner.coordinates.config import CorruptionConfig
from src.utils.geometry.triangulation import triangulate_multiview

FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class CorruptedTrajectory:
    uv_px: FloatArray  # V,T,2; zero where missing
    missing_2d: BoolArray  # V,T; true means missing
    xyz_m: FloatArray  # T,3; zero where missing
    missing_3d: BoolArray
    intervals: NDArray[np.int64]  # selected event, inclusive start, exclusive end
    isolated: BoolArray  # V,T; excludes event gaps and out-of-frame
    noise_px: FloatArray  # before masking, retained for audits only


def coordinate_noise(shape: tuple[int, ...], config: CorruptionConfig, rng: np.random.Generator) -> FloatArray:
    """Isotropic small jitter + Gaussian outliers with population radial P95.

    The Rayleigh tail at R is exp(-R²/2σ²). Solve the two-component survival
    function exactly for the broad component at R=config.noise_p95_px.
    """
    radius = config.noise_p95_px
    small_tail = np.exp(-0.5 * (radius / config.jitter_sigma_px) ** 2) if config.jitter_sigma_px else 0.0
    broad_tail = (0.05 - (1 - config.outlier_probability) * small_tail) / config.outlier_probability
    broad_sigma = radius / np.sqrt(-2 * np.log(broad_tail))
    outlier = rng.random(shape) < config.outlier_probability
    sigma = np.where(outlier, broad_sigma, config.jitter_sigma_px)
    return cast(FloatArray, (rng.standard_normal((*shape, 2)) * sigma[..., None]).astype(np.float32))


def corrupt_trajectory(
    uv_px: FloatArray,
    visible: BoolArray,
    projections: NDArray[np.floating],
    events: NDArray[np.integer],
    *,
    config: CorruptionConfig,
    seed: int,
    noise_enabled: bool = True,
) -> CorruptedTrajectory:
    """Synchronize event gaps across views; sample isolated drops independently.

    Random streams are separate, so rate ablations have nested event selection
    and identical noise/widths/isolated proposals. Overlapping gaps are unioned.
    Event metadata is never returned as a model feature.
    """
    if uv_px.ndim != 3 or uv_px.shape[-1] != 2 or visible.shape != uv_px.shape[:-1]:
        raise ValueError("Require V,T,2 coordinates and V,T visibility")
    frames = uv_px.shape[1]
    if events.shape != (frames,) or not np.isfinite(uv_px).all():
        raise ValueError("Require finite coordinates and one event bitmask per frame")
    event_rng, noise_rng, isolated_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    indices = np.flatnonzero(events)
    selected = event_rng.random(len(indices)) < config.event_probability
    left = event_rng.integers(config.gap_min, config.gap_max + 1, len(indices))
    # Sample right uniformly among the other widths: midpoint is never the event.
    right = event_rng.integers(config.gap_min, config.gap_max, len(indices))
    right += right >= left
    blocked = np.zeros(frames, dtype=bool)
    intervals = []
    for event, before, after in zip(indices[selected], left[selected], right[selected], strict=True):
        start, end = max(0, int(event - before)), min(frames, int(event + after + 1))
        blocked[start:end] = True
        intervals.append((int(event), start, end))
    isolated = (isolated_rng.random(visible.shape) < config.isolated_probability) & visible & ~blocked[None]
    missing = ~visible | blocked[None] | isolated
    noise = coordinate_noise(visible.shape, config, noise_rng) if noise_enabled else np.zeros_like(uv_px)
    observation = np.where(missing[..., None], 0, uv_px + noise).astype(np.float32)
    result = triangulate_multiview(
        observation.transpose(1, 0, 2), (~missing).T.astype(np.float64), projections,
        min_score=0.5, refinement_steps=config.triangulation_steps,
    )
    return CorruptedTrajectory(
        observation, missing, np.where(result.valid[:, None], result.points, 0).astype(np.float32),
        ~result.valid, np.asarray(intervals, dtype=np.int64).reshape(-1, 3), isolated, noise,
    )
