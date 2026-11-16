"""Distinguish measured KP locations from censored or colliding heatmap peaks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KeypointSupport:
    eligible: NDArray[np.bool_]
    boundary: NDArray[np.bool_]
    duplicate: NDArray[np.bool_]
    censor_direction: NDArray[np.int8]
    correlation_groups: NDArray[np.int64]

    def residual(self, projected: NDArray[np.floating], observed: NDArray[np.floating]) -> NDArray[np.float64]:
        """Boundary peaks constrain inward motion, never pull an outside point in."""
        delta = np.asarray(projected - observed, dtype=np.float64)
        return np.where(self.censor_direction < 0, np.maximum(delta, 0),
                        np.where(self.censor_direction > 0, np.minimum(delta, 0), delta))

    def diagnostics(self) -> dict[str, object]:
        return {
            "eligible_indices": np.flatnonzero(self.eligible).tolist(),
            "boundary_censored_indices": np.flatnonzero(self.boundary).tolist(),
            "duplicate_peak_indices": np.flatnonzero(self.duplicate).tolist(),
            "censor_direction_xy": self.censor_direction.tolist(),
            "correlation_groups": self.correlation_groups.tolist(),
        }


def keypoint_support(
    observed: NDArray[np.floating],
    scores: NDArray[np.floating],
    *,
    image_size_hw: tuple[int, int],
    native_size_hw: tuple[int, int],
    min_score: float,
    boundary_margin_cells: float,
    duplicate_distance_cells: float,
) -> KeypointSupport:
    """Keep raw evidence unchanged; exclude unreliable *equality constraints*.

    A peak within the declared native-grid border lacks support on both sides.
    It is censored, not a measured off-image intersection. Two semantic channels
    within one localization cell cannot count as independent correspondences;
    neither is chosen by confidence. Final H projections may still recover such
    points, including genuine intersections on or outside the image boundary.
    """
    points = np.asarray(observed, dtype=np.float64)
    quality = np.asarray(scores, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or quality.shape != (len(points),):
        raise ValueError("KP coordinates/scores must have shapes (N,2)/(N,)")
    if min(*image_size_hw, *native_size_hw) < 2:
        raise ValueError("KP evidence requires image/native dimensions >= 2")
    if not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
        raise ValueError("KP scores must be finite in [0,1]")
    for value in (boundary_margin_cells, duplicate_distance_cells):
        if not np.isfinite(value) or value < 0:
            raise ValueError("KP cell distances must be finite and nonnegative")
    if not np.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("KP min_score must be in [0,1]")
    height, width = image_size_hw
    native_height, native_width = native_size_hw
    scale = np.array([(native_width - 1) / (width - 1), (native_height - 1) / (height - 1)])
    grid = points * scale
    finite = np.isfinite(grid).all(axis=1) & (quality >= min_score)
    boundary = finite & (
        (grid[:, 0] < boundary_margin_cells)
        | (grid[:, 1] < boundary_margin_cells)
        | (grid[:, 0] > native_width - 1 - boundary_margin_cells)
        | (grid[:, 1] > native_height - 1 - boundary_margin_cells)
    )
    distance = np.linalg.norm(grid[:, None] - grid[None], axis=-1)
    close = (distance <= duplicate_distance_cells) & finite[:, None] & finite[None]
    np.fill_diagonal(close, False)
    duplicate = close.any(axis=1)
    direction = np.zeros_like(points, dtype=np.int8)
    for axis, extent in enumerate((native_width, native_height)):
        direction[finite & (grid[:, axis] < boundary_margin_cells), axis] = -1
        direction[finite & (grid[:, axis] > extent - 1 - boundary_margin_cells), axis] = 1
    groups: NDArray[np.int64] = np.arange(len(points), dtype=np.int64)
    # Connected components, including transitive collisions, are one source of
    # localization evidence. A hypothesis may use at most one member per group.
    for a, b in zip(*np.nonzero(close), strict=True):
        groups[groups == groups[b]] = groups[a]
    return KeypointSupport(finite & ~boundary & ~duplicate, boundary, duplicate, direction, groups)
