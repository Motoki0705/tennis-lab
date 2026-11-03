"""Explicit spatial, robust-likelihood and temporal budgets."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DenseLikelihoodConfig:
    interpolation: str = "bicubic_logit"
    initialization: str = "candidates"
    temperature: float = 1.0
    background: float = 0.05
    # Overrides are keyed by camera identity, never by the current LOVO slot.
    view_temperatures: tuple[tuple[str, float], ...] = ()
    view_backgrounds: tuple[tuple[str, float], ...] = ()
    support_logit_drop: float = 5.0
    min_peak_probability: float = 0.05
    x_bounds: tuple[float, float] = (-12.0, 12.0)
    y_bounds: tuple[float, float] = (-24.0, 24.0)
    z_bounds: tuple[float, float] = (-0.2, 12.0)
    coarse_spacing_m: float = 2.0
    coarse_levels: int = 3
    coarse_keep: int = 16
    multistart: int = 2
    max_iterations: int = 40
    min_ray_angle_deg: float = 1.0
    temporal_weight: float = 0.0
    acceleration_scale_mps2: float = 100.0
    event_acceleration_mps2: float = 350.0
    window_size: int = 9

    def __post_init__(self) -> None:
        if self.interpolation not in {"bicubic_logit", "quadratic_logit", "quadratic_log"}:
            raise ValueError("Unknown dense interpolation")
        if self.initialization not in {"candidates", "coarse"}:
            raise ValueError("Unknown dense initialization")
        positive = (self.temperature, self.support_logit_drop, self.coarse_spacing_m,
                    self.acceleration_scale_mps2, self.event_acceleration_mps2)
        if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Dense scales/temperature must be finite and positive")
        if isinstance(self.background, bool) or not 0 < self.background < 1:
            raise ValueError("Dense background must be in (0,1)")
        if not 0 < self.min_peak_probability < 1 or not 0 < self.min_ray_angle_deg < 90:
            raise ValueError("Invalid dense support/ray limits")
        if isinstance(self.temporal_weight, bool) or not math.isfinite(self.temporal_weight) or self.temporal_weight < 0:
            raise ValueError("Dense temporal weight must be finite and nonnegative")
        for bounds in self.bounds:
            if len(bounds) != 2 or any(not math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
                raise ValueError("Invalid dense court bounds")
        for value in (self.coarse_levels, self.coarse_keep, self.multistart, self.max_iterations, self.window_size):
            if type(value) is not int or value < 1:
                raise ValueError("Dense search budgets must be positive integers")
        if self.window_size < 3:
            raise ValueError("Dense temporal window needs at least three frames")
        # Avoid accidentally constructing a huge volume on a small CPU host.
        cells = math.prod(math.ceil((hi - lo) / self.coarse_spacing_m) + 1 for lo, hi in self.bounds)
        if cells > 200_000:
            raise ValueError("Dense coarse volume exceeds 200000 points")
        for rows, temperature in ((self.view_temperatures, True), (self.view_backgrounds, False)):
            if len({name for name, _ in rows}) != len(rows):
                raise ValueError("Duplicate dense camera override")
            for name, camera_value in rows:
                if not name or isinstance(camera_value, bool) or not math.isfinite(camera_value) or camera_value <= 0 or (not temperature and camera_value >= 1):
                    raise ValueError("Invalid dense camera temperature/background")

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return self.x_bounds, self.y_bounds, self.z_bounds

    def for_view(self, camera_id: str) -> tuple[float, float]:
        return (dict(self.view_temperatures).get(camera_id, self.temperature),
                dict(self.view_backgrounds).get(camera_id, self.background))
