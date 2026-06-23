"""Pixel <-> normalized keypoint conversions (numpy)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def clamp_pixel_coordinate(value: float, axis_size: int) -> float:
    """Clamp one coordinate to a zero-based pixel axis."""
    return float(min(max(value, 0.0), max(axis_size - 1, 0)))


def normalize_keypoints(
    keypoints: NDArray[np.float32],
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Normalize pixel keypoints ``(..., 2)`` to the ``[0, 1]`` range.

    Returns a copy; the input is not modified.
    """
    result = keypoints.copy()
    result[..., 0] /= width
    result[..., 1] /= height
    return result


def denormalize_keypoints(
    keypoints: NDArray[np.float32],
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Denormalize ``[0, 1]`` keypoints ``(..., 2)`` back to pixel coordinates.

    Returns a copy; the input is not modified.
    """
    result = keypoints.copy()
    result[..., 0] *= width
    result[..., 1] *= height
    return result


__all__ = [
    "clamp_pixel_coordinate",
    "denormalize_keypoints",
    "normalize_keypoints",
]
