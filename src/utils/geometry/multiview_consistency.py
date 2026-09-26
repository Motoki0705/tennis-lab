"""Per-point geometric consistency of multi-view observations under given cameras.

Every point is triangulated by one unweighted DLT over *all* of its observing
views (no inlier selection), so a camera whose pose is wrong for the point
cannot be voted out. That is the property a camera-pose hypothesis test needs:
the score measures how well the supplied cameras explain every observation.
Reconstruction that must tolerate outlier views uses
``triangulation.triangulate_points`` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.triangulation import PinholeCamera, solve_homogeneous_dlt


@dataclass(frozen=True)
class ConsistencyBounds:
    """Physical plausibility of a triangulated point, in court metres."""

    height_range_m: tuple[float, float]
    max_abs_xy_m: float = 40.0
    min_ray_angle_deg: float = 1.0

    def __post_init__(self) -> None:
        low, high = self.height_range_m
        if not all(math.isfinite(x) for x in (low, high, self.max_abs_xy_m)) or low >= high or self.max_abs_xy_m <= 0:
            raise ValueError("Consistency bounds must be finite with a nonempty height range")
        if not 0 < self.min_ray_angle_deg < 90:
            raise ValueError("Minimum ray angle must be in (0, 90) degrees")


@dataclass(frozen=True)
class PointConsistency:
    """``cost``: mean over observing views of ``min((error/threshold)^2, 1)``; 1 for an
    implausible point. ``support``: plausible and every observing view within threshold.
    ``xyz`` is the DLT point (meaningless where ``plausible`` is false)."""

    xyz: NDArray[np.float64]  # (N,3)
    plausible: NDArray[np.bool_]  # (N,)
    cost: NDArray[np.float64]  # (N,)
    support: NDArray[np.bool_]  # (N,)


def score_multiview_points(
    uv_px: NDArray[np.floating],
    observed: NDArray[np.bool_],
    cameras: tuple[PinholeCamera, ...],
    *,
    threshold_px: float,
    bounds: ConsistencyBounds,
) -> PointConsistency:
    """Score ``(V,N,2)`` pixel observations; every point needs at least two observing views.

    A point is implausible when the DLT is degenerate, it lies behind an
    observing camera, no observing pair has rays separated by
    ``min_ray_angle_deg``, or it leaves the court-metre bounds.
    """
    views = len(cameras)
    if uv_px.ndim != 3 or uv_px.shape[0] != views or uv_px.shape[-1] != 2 or observed.shape != uv_px.shape[:-1] or observed.dtype != np.bool_:
        raise ValueError("Expected (V,N,2) pixels and (V,N) boolean observations matching the cameras")
    if views < 2 or not math.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("Consistency scoring needs two cameras and a positive pixel threshold")
    if (observed.sum(0) < 2).any():
        raise ValueError("Every scored point needs at least two observing views")
    uv = np.asarray(uv_px, np.float64)
    if not np.isfinite(uv[observed]).all():
        raise ValueError("Observed pixels must be finite")
    matrices = np.stack([c.matrix for c in cameras])
    design = np.stack((uv[..., 0, None] * matrices[:, None, 2] - matrices[:, None, 0],
                       uv[..., 1, None] * matrices[:, None, 2] - matrices[:, None, 1]), -2)
    design = np.where(observed[..., None, None], design, 0.)
    xyz, plausible = solve_homogeneous_dlt(design.transpose(1, 0, 2, 3).reshape(uv.shape[1], -1, 4))
    error = np.zeros(observed.shape, np.float64)
    for view, camera in enumerate(cameras):
        projected, front = camera.project(xyz)
        plausible &= ~observed[view] | front
        error[view] = np.linalg.norm(projected - uv[view], axis=-1)
    rays = xyz[None] - np.stack([c.center for c in cameras])[:, None]
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
    separated: NDArray[np.bool_] = np.zeros(len(xyz), bool)
    cosine = math.cos(math.radians(bounds.min_ray_angle_deg))
    for a, b in combinations(range(views), 2):
        separated |= observed[a] & observed[b] & (np.abs((rays[a] * rays[b]).sum(-1)) <= cosine)
    plausible &= separated
    plausible &= (np.abs(xyz[:, :2]) <= bounds.max_abs_xy_m).all(-1)
    plausible &= (xyz[:, 2] >= bounds.height_range_m[0]) & (xyz[:, 2] <= bounds.height_range_m[1])
    cost = (np.minimum((error / threshold_px) ** 2, 1.) * observed).sum(0) / observed.sum(0)
    cost[~plausible] = 1.
    support = plausible & ((error <= threshold_px) | ~observed).all(0)
    return PointConsistency(xyz, plausible, cost, support)
