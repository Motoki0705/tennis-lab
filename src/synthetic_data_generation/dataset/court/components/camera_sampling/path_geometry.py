"""Task-local closed-path geometry for strict Court V4 trajectories."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitTrajectorySpecV4,
    PathConstructorV4,
    PathFamilyV4,
    VerticalProfileV4,
)


def closed_path_points_local(
    trajectory: OrbitTrajectorySpecV4,
    fractions: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Evaluate one closed V4 path in its local metric frame."""
    parameter = _fractions(fractions)
    if trajectory.shape is PathFamilyV4.FREE_SPACE_CYCLE:
        controls = trajectory.control_points_local_m
        if controls is None:  # pragma: no cover - strict V4 contract excludes this
            raise ValueError("Free-space cycle is missing control points.")
        points = closed_polyline_points(
            np.asarray(controls, dtype=np.float64),
            parameter,
        )
        cosine = math.cos(trajectory.orientation_radians)
        sine = math.sin(trajectory.orientation_radians)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
        points[:, :2] = points[:, :2] @ rotation.T
        points[:, 2] += trajectory.base_height_m
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
            raise ValueError(
                "V4 free-space cycle produces non-finite or non-positive points."
            )
        return points
    if trajectory.constructor is PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE:
        anchor = trajectory.anchor_provenance
        if anchor is None:  # pragma: no cover - strict V4 contract excludes this
            raise ValueError("Anchored rounded rectangle is missing provenance.")
        points = rounded_rectangle_points_local(
            center_local_m=anchor.anchor_center_local_m,
            half_width_m=anchor.half_width_m,
            half_height_m=anchor.half_height_m,
            corner_radius_m=anchor.corner_radius_m,
            orientation_radians=anchor.orientation_radians,
            vertical_profile=anchor.vertical_profile,
            vertical_phase_offsets_m=trajectory.vertical_phase_offsets_m,
            fractions=parameter,
        )
        reference_fractions = np.arange(
            len(anchor.reference_points_local_m), dtype=np.float64
        ) / len(anchor.reference_points_local_m)
        reference = rounded_rectangle_points_local(
            center_local_m=anchor.anchor_center_local_m,
            half_width_m=anchor.half_width_m,
            half_height_m=anchor.half_height_m,
            corner_radius_m=anchor.corner_radius_m,
            orientation_radians=anchor.orientation_radians,
            vertical_profile=anchor.vertical_profile,
            vertical_phase_offsets_m=trajectory.vertical_phase_offsets_m,
            fractions=reference_fractions,
        )
        if not np.allclose(
            reference,
            np.asarray(anchor.reference_points_local_m, dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Anchored rounded-rectangle reference points disagree with geometry."
            )
        return points
    if trajectory.shape is PathFamilyV4.CIRCLE:
        xy = _ellipse_xy(
            trajectory.radius_x_m,
            trajectory.radius_x_m,
            parameter,
        )
    elif trajectory.shape is PathFamilyV4.ELLIPSE:
        xy = _ellipse_xy(
            trajectory.radius_x_m,
            trajectory.radius_y_m,
            parameter,
        )
    elif trajectory.shape is PathFamilyV4.ROUNDED_RECTANGLE:
        corner = trajectory.corner_radius_m
        if corner is None:  # pragma: no cover - V4 contract excludes this state
            raise ValueError("Rounded rectangle is missing its corner radius.")
        xy, _tangent = rounded_rectangle_xy_and_tangent(
            half_width_m=trajectory.radius_x_m,
            half_height_m=trajectory.radius_y_m,
            corner_radius_m=corner,
            fractions=parameter,
        )
    else:  # pragma: no cover - strict enum construction is exhaustive
        raise ValueError(f"Unsupported V4 path shape: {trajectory.shape!r}.")
    cosine = math.cos(trajectory.orientation_radians)
    sine = math.sin(trajectory.orientation_radians)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    rotated = xy @ rotation.T
    height = _vertical_height(trajectory, parameter)
    points = np.column_stack((rotated, height))
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise ValueError("V4 path produces non-finite or non-positive camera height.")
    return points


def closed_polyline_points(
    control_points: NDArray[np.floating],
    fractions: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Evaluate an exact closed 3-D control polyline by arc-length fraction."""
    controls = np.asarray(control_points, dtype=np.float64)
    parameter = _fractions(fractions)
    if (
        controls.ndim != 2
        or controls.shape[1] != 3
        or len(controls) < 8
        or not np.isfinite(controls).all()
    ):
        raise ValueError("control_points must contain at least eight finite 3-vectors.")
    closed = np.vstack((controls, controls[:1]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(lengths <= 1.0e-9):
        raise ValueError("Closed control-polyline edges must have positive length.")
    cumulative = np.concatenate((np.zeros(1), np.cumsum(lengths)))
    distances = parameter * cumulative[-1]
    edge = np.searchsorted(cumulative[1:], distances, side="right")
    edge = np.minimum(edge, len(controls) - 1)
    local_distance = distances - cumulative[edge]
    blend = local_distance / lengths[edge]
    result = controls[edge] + (closed[edge + 1] - controls[edge]) * blend[:, None]
    return np.asarray(result, dtype=np.float64)


def rounded_rectangle_xy_and_tangent(
    *,
    half_width_m: float,
    half_height_m: float,
    corner_radius_m: float,
    fractions: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a CCW rounded rectangle at exact horizontal arc-length fractions."""
    parameter = _fractions(fractions)
    half_width = _positive(half_width_m, name="half_width_m")
    half_height = _positive(half_height_m, name="half_height_m")
    radius = _positive(corner_radius_m, name="corner_radius_m")
    if radius >= min(half_width, half_height):
        raise ValueError("corner_radius_m must be smaller than both half extents.")
    horizontal = 2.0 * (half_width - radius)
    vertical = 2.0 * (half_height - radius)
    quarter = 0.5 * math.pi * radius
    lengths = (horizontal, quarter, vertical, quarter) * 2
    perimeter = float(sum(lengths))
    distance = np.mod(parameter, 1.0) * perimeter
    xy: NDArray[np.float64] = np.empty((len(parameter), 2), dtype=np.float64)
    tangent = np.empty_like(xy)
    boundaries = np.cumsum(np.asarray((0.0, *lengths), dtype=np.float64))
    for index, value in enumerate(distance):
        segment = min(int(np.searchsorted(boundaries[1:], value, side="right")), 7)
        local = value - boundaries[segment]
        if segment == 0:
            xy[index] = (-half_width + radius + local, -half_height)
            tangent[index] = (1.0, 0.0)
        elif segment == 1:
            angle = -0.5 * math.pi + local / radius
            xy[index] = (
                half_width - radius + radius * math.cos(angle),
                -half_height + radius + radius * math.sin(angle),
            )
            tangent[index] = (-math.sin(angle), math.cos(angle))
        elif segment == 2:
            xy[index] = (half_width, -half_height + radius + local)
            tangent[index] = (0.0, 1.0)
        elif segment == 3:
            angle = local / radius
            xy[index] = (
                half_width - radius + radius * math.cos(angle),
                half_height - radius + radius * math.sin(angle),
            )
            tangent[index] = (-math.sin(angle), math.cos(angle))
        elif segment == 4:
            xy[index] = (half_width - radius - local, half_height)
            tangent[index] = (-1.0, 0.0)
        elif segment == 5:
            angle = 0.5 * math.pi + local / radius
            xy[index] = (
                -half_width + radius + radius * math.cos(angle),
                half_height - radius + radius * math.sin(angle),
            )
            tangent[index] = (-math.sin(angle), math.cos(angle))
        elif segment == 6:
            xy[index] = (-half_width, half_height - radius - local)
            tangent[index] = (0.0, -1.0)
        else:
            angle = math.pi + local / radius
            xy[index] = (
                -half_width + radius + radius * math.cos(angle),
                -half_height + radius + radius * math.sin(angle),
            )
            tangent[index] = (-math.sin(angle), math.cos(angle))
    return xy, tangent


def rounded_rectangle_points_local(
    *,
    center_local_m: tuple[float, float, float],
    half_width_m: float,
    half_height_m: float,
    corner_radius_m: float,
    orientation_radians: float,
    vertical_profile: VerticalProfileV4,
    vertical_phase_offsets_m: tuple[float, ...],
    fractions: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Evaluate a translated genuine rounded rectangle with an optional lift."""
    parameter = _fractions(fractions)
    center = np.asarray(center_local_m, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_local_m must be one finite 3-vector.")
    xy, _tangent = rounded_rectangle_xy_and_tangent(
        half_width_m=half_width_m,
        half_height_m=half_height_m,
        corner_radius_m=corner_radius_m,
        fractions=parameter,
    )
    orientation = float(orientation_radians)
    if not math.isfinite(orientation):
        raise ValueError("orientation_radians must be finite.")
    cosine = math.cos(orientation)
    sine = math.sin(orientation)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    rotated = xy @ rotation.T + center[:2]
    if vertical_profile is VerticalProfileV4.PLANAR:
        if tuple(vertical_phase_offsets_m) != (0.0,):
            raise ValueError("Planar anchored geometry requires phase offsets [0.0].")
        height = np.full_like(parameter, center[2])
    elif vertical_profile is VerticalProfileV4.RAISED_PHASES:
        height = center[2] + raised_phase_offsets_m(
            vertical_phase_offsets_m,
            parameter,
        )
    else:
        raise ValueError(
            "Anchored rounded rectangles require planar or raised_phases."
        )
    points = np.column_stack((rotated, height))
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise ValueError("Anchored rounded rectangle produced invalid points.")
    return np.asarray(points, dtype=np.float64)


def raised_phase_offsets_m(
    offsets_m: tuple[float, ...],
    fractions: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Evaluate the shared smooth closed raised-phase profile."""
    parameter = _fractions(fractions)
    offsets = np.asarray(offsets_m, dtype=np.float64)
    if (
        offsets.ndim != 1
        or len(offsets) < 4
        or not np.isfinite(offsets).all()
        or np.any(offsets < 0.0)
        or float(np.max(offsets)) <= 0.0
    ):
        raise ValueError("Raised phase offsets must be finite, non-negative and lifted.")
    scaled = np.mod(parameter, 1.0) * len(offsets)
    lower = np.floor(scaled).astype(np.int64) % len(offsets)
    upper = (lower + 1) % len(offsets)
    blend = 0.5 - 0.5 * np.cos(math.pi * (scaled - np.floor(scaled)))
    return np.asarray(
        offsets[lower] * (1.0 - blend) + offsets[upper] * blend,
        dtype=np.float64,
    )


def _ellipse_xy(
    major_m: float,
    minor_m: float,
    fractions: NDArray[np.float64],
) -> NDArray[np.float64]:
    theta = 2.0 * math.pi * fractions
    return np.stack((major_m * np.cos(theta), minor_m * np.sin(theta)), axis=1)


def _vertical_height(
    trajectory: OrbitTrajectorySpecV4,
    fractions: NDArray[np.float64],
) -> NDArray[np.float64]:
    if trajectory.curve_mode is VerticalProfileV4.PLANAR:
        return np.full_like(fractions, trajectory.base_height_m)
    if trajectory.curve_mode is VerticalProfileV4.SINUSOIDAL_HEIGHT:
        theta = 2.0 * math.pi * fractions
        return trajectory.base_height_m + trajectory.vertical_amplitude_m * np.sin(
            trajectory.vertical_cycles * theta + trajectory.vertical_phase_radians
        )
    if trajectory.curve_mode is not VerticalProfileV4.RAISED_PHASES:
        raise ValueError(f"Unsupported V4 vertical mode: {trajectory.curve_mode!r}.")
    result = trajectory.base_height_m + raised_phase_offsets_m(
        trajectory.vertical_phase_offsets_m,
        fractions,
    )
    return np.asarray(result, dtype=np.float64)


def _fractions(value: NDArray[np.floating]) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("fractions must be a non-empty finite one-dimensional array.")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("fractions must stay within [0, 1].")
    return result


def _positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


__all__ = [
    "closed_path_points_local",
    "closed_polyline_points",
    "raised_phase_offsets_m",
    "rounded_rectangle_points_local",
    "rounded_rectangle_xy_and_tangent",
]
