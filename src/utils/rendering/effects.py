"""Reusable matplotlib-3D drawing effects for scene visualization.

Primitives shared by the tennis scene renderer (and any other 3D plot):

- ``render_fading_line_3d``: a polyline whose alpha/width ramp up toward the
  most recent point — used for ball and player motion trails.
- ``render_ground_shadow``: a filled disc on the ground plane — a cheap fake
  shadow anchoring objects visually to the court.
- ``render_ground_ring``: an unfilled circle on the ground plane — used for
  bounce-impact markers.
- ``render_impact_ring``: a ground ring that expands and fades with the age
  of an impact — the shared bounce-marker animation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from numpy.typing import NDArray


def render_fading_line_3d(
    ax: Axes3D,
    positions: NDArray[np.float32],
    *,
    color: str | tuple[float, float, float],
    alpha_range: tuple[float, float] = (0.05, 0.9),
    linewidth_range: tuple[float, float] = (1.0, 3.0),
    zorder: int = 4,
) -> Line3DCollection | None:
    """Draw a 3D polyline that fades in toward its last point.

    Consecutive finite points are connected; segments touching a non-finite
    point are skipped, so gaps in the track stay visible as gaps. Alpha and
    linewidth ramp linearly from the first segment to the last.

    Args:
        ax: Target 3D axis.
        positions: Points of shape (T, 3), oldest first.
        color: Base line color; per-segment alpha is applied on top.
        alpha_range: (oldest, newest) segment alpha.
        linewidth_range: (oldest, newest) segment linewidth.
        zorder: Z-order of the collection.

    Returns:
        The added ``Line3DCollection``, or None when fewer than two
        consecutive finite points exist (nothing is drawn).
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"positions must have shape (T, 3), got {pts.shape}")

    finite = cast("NDArray[np.bool_]", np.isfinite(pts).all(axis=1))
    num_segments_total = pts.shape[0] - 1
    segments: list[NDArray[np.float64]] = []
    ramps: list[float] = []
    for t in range(num_segments_total):
        if finite[t] and finite[t + 1]:
            segments.append(pts[t : t + 2])
            ramp = t / (num_segments_total - 1) if num_segments_total > 1 else 1.0
            ramps.append(ramp)
    if not segments:
        return None

    base_rgba = to_rgba(color)
    alpha_lo, alpha_hi = alpha_range
    width_lo, width_hi = linewidth_range
    colors = [
        (base_rgba[0], base_rgba[1], base_rgba[2], alpha_lo + (alpha_hi - alpha_lo) * r)
        for r in ramps
    ]
    widths = [width_lo + (width_hi - width_lo) * r for r in ramps]

    collection = Line3DCollection(
        segments,
        colors=colors,
        linewidths=widths,
        zorder=zorder,
        capstyle="round",
    )
    ax.add_collection3d(collection)
    return collection


def _circle_points(
    center_xy: tuple[float, float],
    radius: float,
    z: float,
    num_points: int,
) -> NDArray[np.float64]:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    x = center_xy[0] + radius * np.cos(theta)
    y = center_xy[1] + radius * np.sin(theta)
    return np.stack([x, y, np.full_like(x, z)], axis=-1)


def render_ground_shadow(
    ax: Axes3D,
    center_xy: tuple[float, float],
    radius: float,
    *,
    color: str | tuple[float, float, float] = "black",
    alpha: float = 0.25,
    z: float = 0.01,
    num_points: int = 24,
    zorder: int = 1,
) -> Poly3DCollection:
    """Draw a filled disc on the ground plane as a fake contact shadow.

    Args:
        ax: Target 3D axis.
        center_xy: Disc centre on the court plane.
        radius: Disc radius in metres. Must be positive.
        color: Fill color.
        alpha: Fill alpha.
        z: Height offset above the ground plane to avoid z-fighting.
        num_points: Number of polygon vertices.
        zorder: Z-order of the collection.

    Returns:
        The added ``Poly3DCollection``.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    ring = _circle_points(center_xy, radius, z, num_points)
    shadow = Poly3DCollection(
        [ring],
        facecolor=color,
        alpha=alpha,
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_collection3d(shadow)
    return shadow


def render_impact_ring(
    ax: Axes3D,
    center_xy: tuple[float, float],
    age: float,
    *,
    color: str | tuple[float, float, float],
    base_radius: float = 0.15,
    growth: float = 0.45,
    linewidth: float = 2.0,
    zorder: int = 5,
) -> None:
    """Draw an expanding, fading ground ring for an impact of a given age.

    Args:
        ax: Target 3D axis.
        center_xy: Impact position on the court plane.
        age: Normalized impact age in ``[0, 1]``: 0 draws a small opaque
            ring, 1 the fully expanded, nearly transparent one.
        color: Ring color.
        base_radius: Ring radius at ``age == 0`` in metres.
        growth: Radius growth over the full age range in metres.
        linewidth: Line width in points.
        zorder: Z-order of the ring.
    """
    if not 0.0 <= age <= 1.0:
        raise ValueError(f"age must be within [0, 1], got {age}")
    render_ground_ring(
        ax,
        center_xy,
        radius=base_radius + growth * age,
        color=color,
        alpha=0.9 * (1.0 - age) + 0.05,
        linewidth=linewidth,
        zorder=zorder,
    )


def render_ground_ring(
    ax: Axes3D,
    center_xy: tuple[float, float],
    radius: float,
    *,
    color: str | tuple[float, float, float],
    alpha: float = 0.8,
    linewidth: float = 2.0,
    z: float = 0.02,
    num_points: int = 48,
    zorder: int = 5,
) -> None:
    """Draw an unfilled circle on the ground plane (bounce-impact marker).

    Args:
        ax: Target 3D axis.
        center_xy: Ring centre on the court plane.
        radius: Ring radius in metres. Must be positive.
        color: Line color.
        alpha: Line alpha.
        linewidth: Line width in points.
        z: Height offset above the ground plane to avoid z-fighting.
        num_points: Number of points approximating the circle.
        zorder: Z-order of the line.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    ring = _circle_points(center_xy, radius, z, num_points)
    ax.plot(
        ring[:, 0],
        ring[:, 1],
        ring[:, 2],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
