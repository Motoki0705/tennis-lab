"""Analytic containment of complete orbit shapes in the captured-camera XY hull.

This is a camera-position heuristic, not a guarantee of 3DGS image quality.
Each centre uses its own metric court plane; height remains config-owned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.synthetic_data_generation.dataset.court.components.camera_sampling.shapes import (
    shape_support,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitTrajectorySpec,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


def bound_trajectory_candidates(
    candidates: Sequence[OrbitTrajectorySpec],
    *,
    centers: Sequence[OrbitCenter],
    cameras: Sequence[SceneCamera],
    margin_m: float,
    expansion_percent: float = 0.0,
) -> tuple[OrbitTrajectorySpec, ...]:
    """Fit each unit-scale shape inside every inward-offset hull half-plane.

    For normal n and shape map A, use the exact support at A.T n.
    The minimum half-plane allowance therefore bounds the entire curve, not
    just sampled vertices. Radius scales in (0, 1] retain nested diversity.
    Expand the hull about its vertex mean by 1 + percent/100, then inset by
    margin_m. The captured-radius cap expands by the same factor. This is a
    linear-distance percentage, not an area percentage or a per-orbit origin.
    Missing planar support and centres outside the effective hull fail explicitly.
    """
    if not math.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("SfM boundary margin must be finite and non-negative.")
    if not math.isfinite(expansion_percent) or expansion_percent < 0.0:
        raise ValueError("SfM boundary expansion percent must be finite and non-negative.")
    expansion = expansion_percent / 100.0
    if len(cameras) < 3:
        raise ValueError("SfM bounds require at least three captured cameras.")
    captured = np.stack([camera.camera_to_scene.matrix()[:3, 3] for camera in cameras])
    hulls = {}
    for center in centers:
        local = center.scene_from_center.inverse().apply(captured)
        try:
            hull = ConvexHull(local[:, :2])
        except QhullError as error:
            raise ValueError(
                f"SfM cameras have no planar support for {center.key()}."
            ) from error
        normals = hull.equations[:, :2]
        # One hull-owned origin, independent of the selected orbit centre.
        # Vertex mean avoids weighting the expansion by capture density.
        origin = local[hull.vertices, :2].mean(axis=0)
        edge_distances = -(normals @ origin + hull.equations[:, 2])
        allowance = -hull.equations[:, 2] + expansion * edge_distances - margin_m
        if np.any(allowance <= 0.0):
            raise ValueError(
                f"Orbit centre {center.key()} is outside the inset SfM boundary."
            )
        hulls[center.key()] = normals, allowance
    bounded = []
    for candidate in candidates:
        if not 0.0 < candidate.radius_scale <= 1.0:
            raise ValueError("SfM-bounded radius scales must be in (0, 1].")
        normals, allowance = hulls[
            candidate.center_kind, candidate.center_court_instance_id
        ]
        angle = candidate.orientation_radians
        rotation = np.array(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
        )
        shape_map = rotation @ np.diag((1.0, candidate.axis_ratio))
        support = shape_support(candidate.shape, normals @ shape_map)
        radius = min(candidate.base_radius_m * (1.0 + expansion), float(np.min(allowance / support)))
        bounded.append(replace(candidate, base_radius_m=radius))
    return tuple(bounded)
