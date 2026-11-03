"""Explicit settings; calibration parameters are supplied, never fitted here."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateGMMConfig:
    top_k: int = 3
    covariance_scale: float = 1.0
    presence_slope: float = 1.0
    presence_bias: float = 0.0
    min_multiview_mass: float = 0.05
    min_variance_cells2: float = 0.04
    max_variance_cells2: float = 25.0
    log_floor: float = 1e-7
    prior_mean_m: tuple[float, float, float] = (0.0, 0.0, 2.0)
    prior_std_m: tuple[float, float, float] = (8.0, 16.0, 6.0)
    max_components: int = 1024
    max_nfev: int = 150
    estimator: str = "map"
    temporal: str = "none"
    acceleration_std_mps2: float = 150.0
    initial_velocity_std_mps: float = 40.0

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

