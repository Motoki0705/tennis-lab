"""Projective clipping primitives for Court-plane dense targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.data.contracts import CourtInstance2D
from src.tasks.court_detection.geometry import (
    compute_template_to_image_homography,
    court_template_xy,
)

BoolArray: TypeAlias = NDArray[np.bool_]
Float64Array: TypeAlias = NDArray[np.float64]
Int32Array: TypeAlias = NDArray[np.int32]

_MIN_NORMALIZED_DEPTH = 1.0e-6


def _ordered_instance_geometry(
    instance: CourtInstance2D,
) -> tuple[Float64Array, BoolArray]:
    if set(instance.physical_indices.tolist()) != set(range(14)):
        raise ValueError("Court target generation requires physical points 0..13.")
    points: Float64Array = np.empty((14, 2), dtype=np.float64)
    in_front: BoolArray = np.zeros(14, dtype=np.bool_)
    for physical, point, front in zip(
        instance.physical_indices.tolist(),
        instance.points_xy.detach().cpu().numpy(),
        instance.point_in_front.detach().cpu().numpy(),
        strict=True,
    ):
        points[int(physical)] = point
        in_front[int(physical)] = front
    return points, in_front


def _clip_polygon(
    vertices: Float64Array,
    *,
    signed_distance: Float64Array,
) -> Float64Array:
    if vertices.shape[0] != signed_distance.shape[0]:
        raise ValueError("Polygon vertices and clipping distances must align.")
    if vertices.shape[0] == 0:
        return vertices
    result: list[Float64Array] = []
    for index, current in enumerate(vertices):
        following = vertices[(index + 1) % vertices.shape[0]]
        current_distance = float(signed_distance[index])
        following_distance = float(signed_distance[(index + 1) % vertices.shape[0]])
        current_inside = current_distance >= 0.0
        following_inside = following_distance >= 0.0
        if current_inside:
            result.append(current)
        if current_inside == following_inside:
            continue
        denominator = current_distance - following_distance
        if abs(denominator) < np.finfo(np.float64).eps:
            raise ValueError("Court polygon clipping encountered a degenerate edge.")
        fraction = current_distance / denominator
        result.append(current + fraction * (following - current))
    if not result:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CourtPlaneRasterizer:
    """Project one Court plane after clipping its positive-depth image footprint."""

    width: int
    height: int
    homography: Float64Array
    image_points: Float64Array
    point_in_front: BoolArray

    @classmethod
    def from_instance(
        cls,
        instance: CourtInstance2D,
        *,
        width: int,
        height: int,
    ) -> CourtPlaneRasterizer | None:
        """Build an oriented projector, or return ``None`` for an all-behind court."""
        if width <= 0 or height <= 0:
            raise ValueError("Court rasterization requires positive image dimensions.")
        image_points, point_in_front = _ordered_instance_geometry(instance)
        if not bool(point_in_front.any()):
            return None
        homography = compute_template_to_image_homography(
            image_points,
            ransac_reproj_threshold=5.0,
        )
        if homography is None:
            raise ValueError(
                f"Court target homography failed for {instance.court_instance_id!r}."
            )
        matrix = np.asarray(homography, dtype=np.float64)
        template = np.asarray(court_template_xy(14), dtype=np.float64)
        template_h = np.concatenate(
            (template, np.ones((template.shape[0], 1), dtype=np.float64)), axis=1
        )
        depths = (template_h @ matrix.T)[:, 2]
        reference_depths = depths[point_in_front]
        orientation = float(np.sign(np.median(reference_depths)))
        if orientation == 0.0:
            raise ValueError(
                f"Court projection depth is ambiguous for {instance.court_instance_id!r}."
            )
        scale = float(np.median(np.abs(reference_depths)))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            raise ValueError(
                f"Court projection depth scale is degenerate for "
                f"{instance.court_instance_id!r}."
            )
        matrix = matrix * (orientation / scale)
        normalized_depths = (template_h @ matrix.T)[:, 2]
        if np.any(normalized_depths[point_in_front] <= 0.0):
            raise ValueError(
                f"Court in-front flags disagree with the projective half-plane for "
                f"{instance.court_instance_id!r}."
            )
        return cls(
            width=width,
            height=height,
            homography=matrix,
            image_points=image_points,
            point_in_front=point_in_front,
        )

    def project_polygon(self, points_xy: NDArray[np.floating]) -> Int32Array | None:
        """Clip a metric polygon to positive depth and the image before projection."""
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError("Court raster polygons must have shape (N>=3, 2).")
        if not np.isfinite(points).all():
            raise ValueError("Court raster polygons must contain finite coordinates.")
        source_h = np.concatenate(
            (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
        )
        clipped = source_h @ self.homography.T
        distance_functions = (
            lambda value: value[:, 2] - _MIN_NORMALIZED_DEPTH,
            lambda value: value[:, 0],
            lambda value: float(self.width - 1) * value[:, 2] - value[:, 0],
            lambda value: value[:, 1],
            lambda value: float(self.height - 1) * value[:, 2] - value[:, 1],
        )
        for distance_function in distance_functions:
            if clipped.shape[0] == 0:
                return None
            clipped = _clip_polygon(
                clipped,
                signed_distance=cast(Float64Array, distance_function(clipped)),
            )
        if clipped.shape[0] < 3:
            return None
        projected = clipped[:, :2] / clipped[:, 2, None]
        projected[:, 0] = np.clip(projected[:, 0], 0.0, float(self.width - 1))
        projected[:, 1] = np.clip(projected[:, 1], 0.0, float(self.height - 1))
        rounded = np.rint(projected).astype(np.int32)
        deduplicated: list[Int32Array] = []
        for point in rounded:
            if not deduplicated or not np.array_equal(point, deduplicated[-1]):
                deduplicated.append(point)
        if len(deduplicated) > 1 and np.array_equal(deduplicated[0], deduplicated[-1]):
            deduplicated.pop()
        if len(deduplicated) < 3:
            return None
        polygon = np.asarray(deduplicated, dtype=np.int32)
        if np.unique(polygon, axis=0).shape[0] < 3:
            return None
        return polygon


__all__ = ["CourtPlaneRasterizer"]
