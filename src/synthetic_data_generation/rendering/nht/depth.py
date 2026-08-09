"""Explicit depth-unit conversion at the public NHT renderer boundary."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def nht_depth_to_metric(
    depth_nht_scene: NDArray[np.generic],
    *,
    nht_scene_units_per_metre: float,
) -> NDArray[np.float32]:
    """Convert finite public NHT depth into metric-scene metres."""
    scale = _depth_scale(nht_scene_units_per_metre)
    if depth_nht_scene.dtype != np.dtype(np.float32):
        raise TypeError("NHT background depth must use float32 dtype.")
    depth = np.asarray(depth_nht_scene, dtype=np.float32)
    if not np.isfinite(depth).all() or np.any(depth < 0.0):
        raise ValueError("NHT background depth must be finite and non-negative.")
    maximum = float(depth.max(initial=np.float32(0.0)))
    return _validated_nht_depth_to_metric(depth, scale=scale, maximum=maximum)


def _depth_scale(nht_scene_units_per_metre: float) -> np.float32:
    if (
        isinstance(nht_scene_units_per_metre, bool)
        or not isinstance(nht_scene_units_per_metre, (int, float))
    ):
        raise TypeError("nht_scene_units_per_metre must be numeric.")
    scale = float(nht_scene_units_per_metre)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("nht_scene_units_per_metre must be finite and positive.")
    scale_float32 = np.float32(scale)
    if not np.isfinite(scale_float32) or scale_float32 <= np.float32(0.0):
        raise ValueError("nht_scene_units_per_metre must be representable as float32.")
    return scale_float32


def _validated_nht_depth_to_metric(
    depth: NDArray[np.float32],
    *,
    scale: np.float32,
    maximum: float,
) -> NDArray[np.float32]:
    """Convert an already validated depth array without rescanning its payload."""
    largest_metric = maximum / float(scale)
    if largest_metric > float(np.finfo(np.float32).max):
        raise ValueError("Metric NHT depth would overflow float32.")
    result = np.asarray(depth / scale, dtype=np.float32)
    result.setflags(write=False)
    return result


__all__ = ["nht_depth_to_metric"]
