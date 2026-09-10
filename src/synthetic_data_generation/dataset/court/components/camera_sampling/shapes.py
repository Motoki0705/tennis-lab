"""Planar orbit boundaries and their exact half-plane support functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.contracts import OrbitShape


def unit_shape_points(
    shape: OrbitShape, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Intersect angular rays with an L2, L4, or rectangular unit boundary.

    A superellipse is |x|^4 + |y|^4 = 1, giving flattened sides and rounded
    corners. A rectangle is max(|x|, |y|) = 1. Scaling by the two half-extents
    and rotating are owned by the caller, identically for every shape.
    """
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    if shape in (OrbitShape.CIRCLE, OrbitShape.ELLIPSE):
        return directions
    if shape is OrbitShape.RECTANGLE:
        radii = np.max(np.abs(directions), axis=-1)
    elif shape is OrbitShape.SUPERELLIPSE:
        radii = np.sum(directions**4, axis=-1) ** 0.25
    else:
        raise ValueError(f"Unsupported orbit shape: {shape!r}.")
    return directions / radii[..., None]


def shape_support(
    shape: OrbitShape, normals_in_shape_frame: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return max(n dot x) over a unit shape, including all unsampled corners.

    The support norm is dual to the boundary norm: L2 -> L2, L4 -> L(4/3),
    and the rectangle's L-infinity -> L1. Input normals already incorporate
    the orbit's rotation and axis ratio.
    """
    if shape in (OrbitShape.CIRCLE, OrbitShape.ELLIPSE):
        return np.linalg.norm(normals_in_shape_frame, axis=-1)
    if shape is OrbitShape.RECTANGLE:
        return np.sum(np.abs(normals_in_shape_frame), axis=-1)
    if shape is OrbitShape.SUPERELLIPSE:
        return np.sum(np.abs(normals_in_shape_frame) ** (4.0 / 3.0), axis=-1) ** 0.75
    raise ValueError(f"Unsupported orbit shape: {shape!r}.")
