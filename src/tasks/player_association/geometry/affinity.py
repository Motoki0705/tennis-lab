"""Geometric evidence that two camera-local track segments are one person.

Two segments of different cameras are compared by the median court-plane
distance of their footpoints over the frames both observe (``ground_distance``).
The score is the log-likelihood ratio of that distance under two models:

* same person: the footpoint discrepancy is isotropic Gaussian noise in the
  court plane, so its norm is Rayleigh with scale ``sigma_m``;
* different people: the other person stands anywhere in an area ``area_m2``
  around the court, so the density of a distance ``d`` is ``2 pi d / area``.

``log(area / (2 pi sigma^2)) - d^2 / (2 sigma^2)`` is positive for small and
negative for large distances. A median over few shared frames is weak
evidence, so the score is scaled by the shared time up to ``full_evidence_s``.
Segments that share no frame carry no geometric evidence (score 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.geometry.footpoints import GroundDistance


@dataclass(frozen=True)
class GeometryAffinityConfig:
    sigma_m: float
    area_m2: float
    full_evidence_s: float
    max_abs_score: float

    def __post_init__(self) -> None:
        if not (self.sigma_m > 0 and self.area_m2 > 2 * math.pi * self.sigma_m ** 2 and self.full_evidence_s > 0 and self.max_abs_score > 0):
            raise ValueError(f"Invalid geometry affinity config: {self}")


def distance_log_likelihood_ratio(distance_m: float, config: GeometryAffinityConfig) -> float:
    """Clipped same-vs-different log-likelihood ratio of a court-plane distance."""
    if not distance_m >= 0:
        raise ValueError(f"Distance must be finite and nonnegative, got {distance_m}")
    ratio = math.log(config.area_m2 / (2 * math.pi * config.sigma_m ** 2)) - distance_m ** 2 / (2 * config.sigma_m ** 2)
    return float(np.clip(ratio, -config.max_abs_score, config.max_abs_score))


def geometry_score(distance: GroundDistance, fps: float, config: GeometryAffinityConfig) -> float:
    """Evidence-weighted log-likelihood ratio; 0 when no frame is shared."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if distance.shared_frames == 0:
        return 0.
    weight = min(1., float(distance.shared_frames) / (config.full_evidence_s * fps))
    return weight * distance_log_likelihood_ratio(distance.median_m, config)


def rayleigh_scale(distances_m: NDArray[np.float64]) -> float:
    """Maximum-likelihood Rayleigh scale ``sqrt(sum d^2 / 2n)`` of same-person distances."""
    values = np.asarray(distances_m, np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Rayleigh fit needs a nonempty 1D array of finite nonnegative distances")
    return float(np.sqrt(np.mean(values ** 2) / 2))
