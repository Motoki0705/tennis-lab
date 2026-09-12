"""Type-preserving court refinement on a fixed metric ground-plane raster.

Court count and scene scale are fixed by the supplied reconstruction alignment.
This experimental method refines placement only; it does not declare production
acceptance or reuse the baseline's acceptance status for its changed transforms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import differential_evolution

from src.synthetic_data_generation.alignment.whole_court import COURT_LINE_SEGMENTS
from src.utils.schema.court import CENTER_MARK_LENGTH, HALF_LENGTH


def typed_court_segments() -> tuple[tuple[int, NDArray[np.float64]], ...]:
    """Six invariant foreground types, including both short center marks."""
    segments = []
    for segment in COURT_LINE_SEGMENTS:
        if segment.name.startswith("doubles_"):
            kind = 1
        elif segment.name.startswith("singles_"):
            kind = 2
        elif segment.name.startswith("baseline_"):
            kind = 0
        elif segment.name.startswith("service_line_"):
            kind = 3
        else:
            kind = 4
        segments.append(
            (kind, np.asarray([segment.start, segment.end], dtype=np.float64))
        )
    for sign in (-1, 1):
        segments.append(
            (
                5,
                np.asarray(
                    [
                        [0, sign * HALF_LENGTH],
                        [0, sign * (HALF_LENGTH - CENTER_MARK_LENGTH)],
                    ],
                    dtype=np.float64,
                ),
            )
        )
    return tuple(segments)


def transform_segments(
    parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Apply metric (center u, center v, yaw) to regulation segments."""
    u, v, yaw = parameters
    rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return tuple(segment @ rotation.T + [u, v] for _, segment in typed_court_segments())


@dataclass(frozen=True)
class SemanticRasterObjective:
    """Equal-type mean support; a baseline cannot match a service-line channel."""

    rasters: NDArray[np.float32]
    bounds_uv: tuple[float, float, float, float]
    spacing: float

    def __post_init__(self) -> None:
        if self.rasters.ndim != 3 or self.rasters.shape[0] not in (1, 6):
            raise ValueError("Expected one binary or six semantic foreground rasters.")
        if not np.isfinite(self.rasters).all() or np.any(self.rasters < 0):
            raise ValueError("Raster evidence must be finite and nonnegative.")
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("Raster spacing must be positive and finite.")
        lo_u, hi_u, lo_v, hi_v = self.bounds_uv
        if not np.isfinite(self.bounds_uv).all() or lo_u >= hi_u or lo_v >= hi_v:
            raise ValueError("Invalid metric raster bounds.")
        expected = (
            int(np.ceil((hi_v - lo_v) / self.spacing)) + 1,
            int(np.ceil((hi_u - lo_u) / self.spacing)) + 1,
        )
        if self.rasters.shape[1:] != expected:
            raise ValueError("Raster shape disagrees with bounds and spacing.")

    def score(self, parameters: NDArray[np.float64]) -> float:
        """Score a placement, giving unseen/out-of-bounds line samples zero support."""
        if parameters.shape != (3,) or not np.isfinite(parameters).all():
            raise ValueError("Placement must be finite (u,v,yaw).")
        samples: list[list[NDArray[np.float64]]] = [[] for _ in range(6)]
        for (kind, _), segment in zip(
            typed_court_segments(), transform_segments(parameters), strict=True
        ):
            count = max(
                3,
                int(np.ceil(np.linalg.norm(segment[1] - segment[0]) / self.spacing))
                + 1,
            )
            samples[kind].append(np.linspace(segment[0], segment[1], count))
        means = []
        for kind, groups in enumerate(samples):
            points = np.concatenate(groups)
            coordinates = np.stack(
                (
                    (points[:, 1] - self.bounds_uv[2]) / self.spacing,
                    (points[:, 0] - self.bounds_uv[0]) / self.spacing,
                )
            )
            raster = self.rasters[0 if len(self.rasters) == 1 else kind]
            means.append(
                float(
                    np.mean(
                        map_coordinates(
                            raster, coordinates, order=1, mode="constant", cval=0.0
                        )
                    )
                )
            )
        return float(np.mean(means))


def refine_placement(
    objective: SemanticRasterObjective,
    initial: NDArray[np.float64],
    *,
    translation_radius_metres: float,
    yaw_radius_radians: float,
    smoothing_metres: float,
    seed: int,
    maximum_iterations: int,
) -> NDArray[np.float64]:
    """Deterministic bounded placement search using fit-camera evidence only."""
    objective.score(initial)
    for value in (translation_radius_metres, yaw_radius_radians, smoothing_metres):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Search radii and smoothing must be positive and finite.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    if not np.any(objective.rasters > 0):
        raise ValueError("Cannot refine placement without foreground evidence.")
    smoothed = SemanticRasterObjective(
        gaussian_filter(
            objective.rasters,
            sigma=(
                0,
                smoothing_metres / objective.spacing,
                smoothing_metres / objective.spacing,
            ),
        ),
        objective.bounds_uv,
        objective.spacing,
    )
    radii = np.asarray(
        [translation_radius_metres, translation_radius_metres, yaw_radius_radians]
    )
    result = differential_evolution(
        lambda parameters: -smoothed.score(parameters),
        list(zip(initial - radii, initial + radii, strict=True)),
        x0=initial,
        seed=seed,
        maxiter=maximum_iterations,
        popsize=12,
        polish=True,
        workers=1,
        updating="immediate",
        tol=1e-6,
    )
    if not result.success:
        raise RuntimeError(f"Placement optimization did not converge: {result.message}")
    if -float(result.fun) + 1e-8 < smoothed.score(initial):
        raise RuntimeError("Placement optimization reduced the initial support.")
    return np.asarray(result.x, dtype=np.float64)
