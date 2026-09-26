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


def normalize_grid_keypoints(
    keypoints: NDArray[np.floating],
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Normalize pixel keypoints ``(..., 2)`` by ``(W-1, H-1)``.

    This is the pixel-grid convention of the court and ball predictors: the
    first and last pixel centres map to 0 and 1. It differs from
    :func:`normalize_keypoints` (division by ``W``/``H``). Returns a float32 copy.
    """
    return (np.asarray(keypoints, np.float32) / _grid_extent(width, height)).astype(np.float32)


def denormalize_grid_keypoints(
    keypoints: NDArray[np.floating],
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Inverse of :func:`normalize_grid_keypoints`. Returns a float32 copy."""
    return (np.asarray(keypoints, np.float32) * _grid_extent(width, height)).astype(np.float32)


def _grid_extent(width: int, height: int) -> NDArray[np.float32]:
    if width < 1 or height < 1:
        raise ValueError(f"Image size must be positive, got {(width, height)}")
    return np.array([max(width - 1, 1), max(height - 1, 1)], np.float32)


__all__ = [
    "clamp_pixel_coordinate",
    "denormalize_grid_keypoints",
    "denormalize_keypoints",
    "normalize_grid_keypoints",
    "normalize_keypoints",
]
