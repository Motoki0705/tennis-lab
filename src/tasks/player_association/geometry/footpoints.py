"""Footpoints of person boxes on the court plane, and their cross-camera distances.

The footpoint is the bottom-centre of the tracked box, back-projected onto the
court plane ``z = 0`` through the side-resolved camera. Ankle keypoints were
rejected: an ankle sits ~0.1 m above the ground, which a low camera (Meiji
cam2) turns into metres of depth error when projected to ``z = 0`` (#933).
A footpoint is invalid, never estimated, when the box is not observed, its
bottom touches the image border (the feet are cut off) or its ray misses the
plane in front of the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.triangulation import PinholeCamera


@dataclass(frozen=True)
class FootpointConfig:
    bottom_border_px: float = 4.

    def __post_init__(self) -> None:
        if not self.bottom_border_px >= 0:
            raise ValueError("bottom_border_px must be nonnegative")


def ground_footpoints(boxes: NDArray[np.floating], observed: NDArray[np.bool_], camera: PinholeCamera, image_height: int,
                      config: FootpointConfig) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Court-plane ``(x, y)`` of ``(..., 4)`` xyxy boxes and their validity ``(...)``."""
    boxes = np.asarray(boxes, np.float64)
    if boxes.shape[-1:] != (4,) or observed.shape != boxes.shape[:-1]:
        raise ValueError("Boxes must be (..., 4) aligned to observed (...)")
    bottom = np.stack(((boxes[..., 0] + boxes[..., 2]) / 2, boxes[..., 3]), -1)
    points, hit = camera.backproject_to_plane(bottom)
    valid = observed & hit & (boxes[..., 3] <= image_height - config.bottom_border_px)
    return np.where(valid[..., None], points[..., :2], 0.), valid


@dataclass(frozen=True)
class GroundDistance:
    median_m: float  # nan when no frame is shared
    shared_frames: int


def ground_distance(points_a: NDArray[np.float64], valid_a: NDArray[np.bool_], points_b: NDArray[np.float64],
                    valid_b: NDArray[np.bool_]) -> GroundDistance:
    """Median court-plane distance over the frames where both footpoints are valid."""
    shared = valid_a & valid_b
    count = int(shared.sum())
    if not count:
        return GroundDistance(float("nan"), 0)
    return GroundDistance(float(np.median(np.linalg.norm(points_a[shared] - points_b[shared], axis=-1))), count)
