"""Project image-space court-line pixels onto a bounded ground plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.components.ground.plane import (
    GroundPlaneEstimate,
)
from src.synthetic_data_generation.scene_contract import SceneCamera

if TYPE_CHECKING:
    from src.synthetic_data_generation.alignment.components.evidence.ground_line_raster import (
        GroundLineMapSettings,
    )


@dataclass(frozen=True)
class ProjectedLinePixels:
    """Valid line pixels intersected with the bounded ground plane."""

    points_scene: NDArray[np.float64]
    points_uv: NDArray[np.float64]
    probabilities: NDArray[np.float32]
    camera_ranges: NDArray[np.float64]
    proximity_weights: NDArray[np.float64]
    input_count: int
    invalid_parallel_count: int
    invalid_behind_count: int
    invalid_range_count: int
    invalid_bounds_count: int


def expanded_plane_bounds(
    plane: GroundPlaneEstimate,
    *,
    margin: float,
) -> tuple[float, float, float, float]:
    """Expand point-supported plane bounds by a fixed scene-unit margin."""
    if margin <= 0.0:
        raise ValueError("margin must be positive.")
    u_min, u_max, v_min, v_max = plane.support_uv_bounds
    return (
        u_min - margin,
        u_max + margin,
        v_min - margin,
        v_max + margin,
    )


def project_line_pixels_to_ground(
    camera: SceneCamera,
    pixels_xy: NDArray[np.floating[Any]],
    probabilities: NDArray[np.floating[Any]],
    *,
    plane: GroundPlaneEstimate,
    bounds: tuple[float, float, float, float],
    settings: GroundLineMapSettings,
) -> ProjectedLinePixels:
    """Back-project original-image pixels and apply proximity weighting."""
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float32)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_xy must have shape (N, 2), got {pixels.shape}.")
    if probability.shape != (len(pixels),):
        raise ValueError("probabilities must have shape (N,).")
    if not np.isfinite(pixels).all() or not np.isfinite(probability).all():
        raise ValueError("Projection inputs must contain only finite values.")
    if bool(np.any(probability < 0.0)) or bool(np.any(probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")

    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(camera.camera_to_scene, dtype=np.float64).reshape(4, 4)
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    directions_camera = homogeneous @ np.linalg.inv(intrinsics).T
    directions_scene = directions_camera @ pose[:3, :3].T
    directions_scene /= np.linalg.norm(directions_scene, axis=1, keepdims=True)
    camera_center = pose[:3, 3]
    normal = np.asarray(plane.normal, dtype=np.float64)
    denominator = directions_scene @ normal
    numerator = -(float(camera_center @ normal) + plane.offset)
    parallel = np.abs(denominator) < settings.min_ray_plane_cosine
    distances = np.divide(
        numerator,
        denominator,
        out=np.full(len(pixels), np.nan, dtype=np.float64),
        where=~parallel,
    )
    behind = distances <= 0.0
    excessive_range = distances > settings.max_ray_distance
    finite_forward = ~(parallel | behind | excessive_range | ~np.isfinite(distances))
    selected_indices = np.flatnonzero(finite_forward)
    selected_ranges = distances[selected_indices]
    points_scene = (
        camera_center + selected_ranges[:, None] * directions_scene[selected_indices]
    )
    points_uv = plane.to_uv(points_scene)
    u_min, u_max, v_min, v_max = bounds
    outside = (
        (points_uv[:, 0] < u_min)
        | (points_uv[:, 0] > u_max)
        | (points_uv[:, 1] < v_min)
        | (points_uv[:, 1] > v_max)
    )
    in_bounds = ~outside
    selected_ranges = selected_ranges[in_bounds]
    weights = 1.0 / (
        1.0
        + np.power(
            selected_ranges / settings.proximity_scale,
            settings.proximity_power,
        )
    )
    return ProjectedLinePixels(
        points_scene=points_scene[in_bounds],
        points_uv=points_uv[in_bounds],
        probabilities=probability[selected_indices[in_bounds]],
        camera_ranges=selected_ranges,
        proximity_weights=weights,
        input_count=len(pixels),
        invalid_parallel_count=int(parallel.sum()),
        invalid_behind_count=int((~parallel & behind).sum()),
        invalid_range_count=int((~parallel & ~behind & excessive_range).sum()),
        invalid_bounds_count=int(outside.sum()),
    )
