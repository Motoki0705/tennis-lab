"""Image evidence and diversity descriptors for projected court lines."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.homography_quality import (
    OUTER_COURT_INDICES,
)

COURT_LINE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
)


def line_edge_support(
    gray_image: NDArray[np.uint8],
    projected_keypoints_normalized: NDArray[np.floating],
    *,
    distance_tolerance_px: float,
    max_side: int,
) -> float:
    """Return the fraction of projected line pixels near a Canny edge."""
    gray = _as_gray_image(gray_image)
    if distance_tolerance_px <= 0.0:
        raise ValueError(
            f"distance_tolerance_px must be positive, got {distance_tolerance_px}."
        )
    if max_side <= 0:
        raise ValueError(f"max_side must be positive, got {max_side}.")
    points = np.asarray(projected_keypoints_normalized, dtype=np.float32)
    if points.shape != (14, 2) or not np.isfinite(points).all():
        raise ValueError(
            f"Expected finite projected keypoints with shape (14, 2), got {points.shape}."
        )

    height, width = gray.shape
    scale = min(1.0, float(max_side) / max(width, height))
    if scale < 1.0:
        gray = np.asarray(
            cv2.resize(
                gray,
                (max(2, round(width * scale)), max(2, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )
    resized_height, resized_width = gray.shape
    edges = cv2.Canny(gray, 50, 140)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    pixel_points = points * np.asarray(
        [resized_width - 1, resized_height - 1],
        dtype=np.float32,
    )

    distances: list[float] = []
    for first_index, second_index in COURT_LINE_EDGES:
        first = pixel_points[first_index]
        second = pixel_points[second_index]
        sample_count = max(2, int(round(float(np.linalg.norm(second - first)))))
        samples = np.linspace(first, second, sample_count)
        x_indices = np.clip(
            np.rint(samples[:, 0]).astype(np.int32),
            0,
            resized_width - 1,
        )
        y_indices = np.clip(
            np.rint(samples[:, 1]).astype(np.int32),
            0,
            resized_height - 1,
        )
        distances.extend(distance[y_indices, x_indices].astype(float).tolist())
    if not distances:
        return 0.0
    return float(np.mean(np.asarray(distances) <= distance_tolerance_px))


def image_diversity_metrics(
    bgr_image: NDArray[np.uint8],
    projected_keypoints_normalized: NDArray[np.floating],
) -> dict[str, float | str]:
    """Compute compact color, brightness, and exact-duplicate descriptors."""
    image = np.asarray(bgr_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 BGR image with shape (H, W, 3), got {image.shape}."
        )
    points = np.asarray(projected_keypoints_normalized, dtype=np.float32)
    resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    polygon = points[OUTER_COURT_INDICES] * np.asarray([127, 127], dtype=np.float32)
    mask: NDArray[np.uint8] = np.zeros((128, 128), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        pixels = hsv.reshape(-1, 3)
    median_hsv = np.median(pixels, axis=0)
    return {
        "dhash64": _difference_hash(image),
        "court_hue_median": float(median_hsv[0]),
        "court_saturation_median": float(median_hsv[1]),
        "brightness_mean": float(np.mean(pixels[:, 2])),
        "surface_color_bucket": _surface_color_bucket(median_hsv),
        "brightness_bucket": _brightness_bucket(float(np.mean(pixels[:, 2]))),
    }


def _as_gray_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 grayscale image with shape (H, W), got {array.shape}."
        )
    if min(array.shape) <= 1:
        raise ValueError(f"Image sides must be greater than one, got {array.shape}.")
    return np.asarray(array, dtype=np.uint8)


def _difference_hash(image: NDArray[np.uint8]) -> str:
    gray = cv2.cvtColor(
        cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    bits = gray[:, 1:] > gray[:, :-1]
    value = sum(int(bit) << index for index, bit in enumerate(bits.ravel()))
    return f"{value:016x}"


def _surface_color_bucket(hsv: NDArray[np.floating]) -> str:
    hue, saturation = float(hsv[0]), float(hsv[1])
    if saturation < 35.0:
        return "neutral_low_saturation"
    if hue < 25.0 or hue >= 165.0:
        return "red_clay_orange"
    if hue < 90.0:
        return "green"
    if hue < 140.0:
        return "blue_cyan"
    return "other"


def _brightness_bucket(brightness: float) -> str:
    if brightness < 90.0:
        return "dark"
    if brightness < 180.0:
        return "mid"
    return "bright"
