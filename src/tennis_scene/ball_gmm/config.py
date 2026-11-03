"""Explicit settings; calibration parameters are supplied, never fitted here."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateGMMConfig:
    top_k: int
    covariance_scale: float
    presence_slope: float
    presence_bias: float
    min_multiview_mass: float
    min_variance_cells2: float
    max_variance_cells2: float
    log_floor: float
    prior_mean_m: tuple[float, float, float]
    prior_std_m: tuple[float, float, float]
    max_components: int
    max_nfev: int
    estimator: str
    temporal: str
    acceleration_std_mps2: float
    initial_velocity_std_mps: float

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k not in (2, 3):
            raise ValueError("Candidate GMM top_k must be 2 or 3")
        for value in (self.covariance_scale, self.presence_slope,
                      self.min_variance_cells2, self.max_variance_cells2,
                      self.acceleration_std_mps2, self.initial_velocity_std_mps):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Candidate GMM scales must be positive finite numbers")
        if (any(type(v) not in (int, float) for v in (self.presence_bias, self.min_multiview_mass, self.log_floor))
                or not math.isfinite(self.presence_bias)
                or not 0 < self.min_multiview_mass <= 1 or not 0 < self.log_floor < 1
                or self.min_variance_cells2 > self.max_variance_cells2):
            raise ValueError("Invalid candidate GMM presence/curvature bounds")
        if any(type(v) is not int or v < 1 for v in (self.max_components, self.max_nfev)):
            raise ValueError("Candidate GMM budgets must be positive integers")
        if (len(self.prior_mean_m) != 3 or len(self.prior_std_m) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (*self.prior_mean_m, *self.prior_std_m))
                or any(v <= 0 for v in self.prior_std_m)):
            raise ValueError("Candidate GMM requires a finite 3D Gaussian prior")
        if self.estimator not in {"map", "mean"} or self.temporal not in {"none", "kalman", "rts"}:
            raise ValueError("Unknown candidate GMM estimator/temporal selection")
