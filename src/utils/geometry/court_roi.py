"""Image region of the court ground plane expanded by explicit margins."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    CourtConfig,
    court_keypoints_3d,
)

__all__ = ["court_footpoint_polygon_px"]


def court_footpoint_polygon_px(
    keypoints_px: np.ndarray,
    *,
    size: tuple[int, int],
    sideline_margin_m: float,
    baseline_margin_m: float,
) -> tuple[tuple[float, float], ...]:
    """Clip the expanded court's visible image region, including large margins.

    Projecting four corners directly can fold the polygon when a margin reaches
    behind the camera. Clipping the image by inverse-homography halfplanes keeps
    the branch containing the court centre and never includes that folded region.
    The expanded rectangle is invariant under the camera-local half-turn, so
    CourtKP14 in physical or camera-local order defines the same image region.

    ``keypoints_px`` are the 14 court ground keypoints in image pixels.
    """
    if keypoints_px.shape != (14, 2) or not np.isfinite(keypoints_px).all():
        raise ValueError("Court footpoint filtering requires 14 finite keypoints.")
    if (
        not np.isfinite(sideline_margin_m)
        or not np.isfinite(baseline_margin_m)
        or sideline_margin_m < 0
        or baseline_margin_m < 0
    ):
        raise ValueError(
            "Court footpoint filter margins must be finite and non-negative."
        )
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Court footpoint filtering requires a positive image size.")
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    pixels: np.ndarray = keypoints_px.astype(np.float32)
    physical_to_pixels, _ = cv2.findHomography(
        physical.astype(np.float32), pixels, method=0
    )
    if physical_to_pixels is None:
        raise ValueError("Court footpoint filter homography fit failed.")
    x = HALF_DOUBLES_WIDTH + sideline_margin_m
    y = HALF_LENGTH + baseline_margin_m
    try:
        inverse = np.linalg.inv(physical_to_pixels)
    except np.linalg.LinAlgError as error:
        raise ValueError("Court footpoint filter homography is singular.") from error
    if not np.isfinite(inverse).all() or abs(physical_to_pixels[2, 2]) < 1e-12:
        raise ValueError("Court footpoint filter has no finite court-centre branch.")
    inverse *= np.sign(physical_to_pixels[2, 2])
    world_x, world_y, denominator = inverse
    halfplanes = (
        denominator,
        x * denominator + world_x,
        x * denominator - world_x,
        y * denominator + world_y,
        y * denominator - world_y,
    )
    polygon = np.array(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )
    for halfplane in halfplanes:
        polygon = _clip_image_halfplane(polygon, halfplane)
        if len(polygon) < 3:
            raise ValueError("Court footpoint filter does not intersect the image.")
    if abs(cv2.contourArea(polygon.astype(np.float32))) < 1e-6:
        raise ValueError("Court footpoint filter has no visible image area.")
    return tuple((float(point[0]), float(point[1])) for point in polygon)


def _clip_image_halfplane(polygon: np.ndarray, halfplane: np.ndarray) -> np.ndarray:
    """Retain pixel points satisfying a*u + b*v + c >= 0."""
    result: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(previous @ halfplane[:2] + halfplane[2])
    for current in polygon:
        current_value = float(current @ halfplane[:2] + halfplane[2])
        if (previous_value >= 0) != (current_value >= 0):
            fraction = previous_value / (previous_value - current_value)
            result.append(previous + fraction * (current - previous))
        if current_value >= 0:
            result.append(current)
        previous, previous_value = current, current_value
    unique: list[np.ndarray] = []
    for point in result:
        if not unique or not np.allclose(point, unique[-1], rtol=0.0, atol=1e-8):
            unique.append(point)
    if len(unique) > 1 and np.allclose(unique[0], unique[-1], rtol=0.0, atol=1e-8):
        unique.pop()
    clipped: np.ndarray = np.asarray(unique, dtype=np.float64).reshape(-1, 2)
    return clipped
