"""Aggregate proximity-weighted projected lines into a ground-plane raster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.components.ground.projection import (
    ProjectedLinePixels,
)


@dataclass(frozen=True)
class GroundLineMapSettings:
    """Frozen probability, projection, weighting, and raster settings."""

    probability_threshold: float = 0.5
    proximity_scale: float = 0.35
    proximity_power: float = 2.0
    min_ray_plane_cosine: float = 0.05
    max_ray_distance: float = 3.0
    bounds_margin: float = 0.05
    grid_spacing: float = 0.0025
    min_projected_pixels: int = 20

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must lie in [0, 1].")
        for name, value in (
            ("proximity_scale", self.proximity_scale),
            ("proximity_power", self.proximity_power),
            ("min_ray_plane_cosine", self.min_ray_plane_cosine),
            ("max_ray_distance", self.max_ray_distance),
            ("bounds_margin", self.bounds_margin),
            ("grid_spacing", self.grid_spacing),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.min_ray_plane_cosine >= 1.0:
            raise ValueError("min_ray_plane_cosine must be smaller than one.")
        if (
            isinstance(self.min_projected_pixels, bool)
            or self.min_projected_pixels <= 0
        ):
            raise ValueError("min_projected_pixels must be a positive integer.")


class GroundLineAccumulator:
    """Accumulate at most one proximity-weighted contribution per view/cell."""

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        grid_spacing: float,
    ) -> None:
        u_min, u_max, v_min, v_max = bounds
        if u_min >= u_max or v_min >= v_max:
            raise ValueError("Ground-line raster bounds must have positive area.")
        if grid_spacing <= 0.0:
            raise ValueError("grid_spacing must be positive.")
        self.bounds = tuple(float(value) for value in bounds)
        self.grid_spacing = float(grid_spacing)
        self.width = int(np.ceil((u_max - u_min) / grid_spacing)) + 1
        self.height = int(np.ceil((v_max - v_min) / grid_spacing)) + 1
        self.evidence_sum: NDArray[np.float32] = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )
        self.weight_sum: NDArray[np.float32] = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )
        self.view_count: NDArray[np.uint16] = np.zeros(
            (self.height, self.width),
            dtype=np.uint16,
        )

    def add_view(self, projection: ProjectedLinePixels) -> int:
        """Rasterize one view with a max reducer, then add it globally."""
        if len(projection.points_uv) == 0:
            return 0
        u_min, _, v_min, _ = self.bounds
        columns = np.rint(
            (projection.points_uv[:, 0] - u_min) / self.grid_spacing
        ).astype(np.int64)
        rows = np.rint((projection.points_uv[:, 1] - v_min) / self.grid_spacing).astype(
            np.int64
        )
        valid = (
            (columns >= 0) & (columns < self.width) & (rows >= 0) & (rows < self.height)
        )
        flat_indices = rows[valid] * self.width + columns[valid]
        if len(flat_indices) == 0:
            return 0
        evidence = (
            projection.probabilities[valid].astype(np.float64)
            * projection.proximity_weights[valid]
        )
        weights = projection.proximity_weights[valid]
        view_evidence: NDArray[np.float32] = np.zeros(
            self.height * self.width,
            dtype=np.float32,
        )
        view_weight: NDArray[np.float32] = np.zeros(
            self.height * self.width,
            dtype=np.float32,
        )
        np.maximum.at(view_evidence, flat_indices, evidence.astype(np.float32))
        np.maximum.at(view_weight, flat_indices, weights.astype(np.float32))
        view_mask = view_weight > 0.0
        self.evidence_sum.ravel()[view_mask] += view_evidence[view_mask]
        self.weight_sum.ravel()[view_mask] += view_weight[view_mask]
        self.view_count.ravel()[view_mask] += 1
        return int(view_mask.sum())

    def arrays(self) -> dict[str, NDArray[Any]]:
        """Return aggregate evidence, weight, support, and normalized score."""
        mean_probability = np.divide(
            self.evidence_sum,
            self.weight_sum,
            out=np.zeros_like(self.evidence_sum),
            where=self.weight_sum > 0.0,
        )
        return {
            "evidence_sum": self.evidence_sum,
            "weight_sum": self.weight_sum,
            "view_count": self.view_count,
            "mean_probability": mean_probability,
        }
