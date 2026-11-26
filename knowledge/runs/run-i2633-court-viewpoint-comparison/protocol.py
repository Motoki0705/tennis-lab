"""Prespecified viewpoint and scoring rules for the court comparison."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import cv2
import numpy as np

from src.tasks.court_detection.geometry.confidence_homography import (
    estimate_confidence_homography,
)
from src.tasks.court_detection.geometry.homography import court_template_xy
from src.tasks.court_detection.scaling.real_split import video_id

SELECTION_SEED = 42
TRAIN_SEEDS = (42, 43)
CONDITIONS = ("narrow", "broad")
UPDATES = 6000
BATCH_SIZE = 8
LONG_SIDE = 512
REAL_VERSION = "real-video-v1"
REAL_REGISTRY_SHA256 = (
    "d943d569c87e0974590eeaf01cdc1b526da4f6eeb0ca65bbd54c05a12c32d60a"
)
REAL_SPLIT_COUNTS = {"val": (4468, 249), "test": (4372, 249)}

CAPTURE_LIMITS = np.array([3.0, 1.0, 15.0, 12.0])  # xy m, z m, axis deg, HFOV deg
LOW_ANGLE_DEG = 12.0
OBLIQUE_DEG = 5.0
SIDE_ON_DEG = 45.0
MIN_DEPTH_SPAN = 1e-6  # below this relative variation, the depth direction is undefined
H_MIN_SCORE = 0.05
H_INLIER_RATIO = 0.005
H_ACCURACY_RATIO = 0.01

COURT = court_template_xy().astype(np.float64)
CENTERED_COURT = COURT - COURT.mean(axis=0)
HOMOGENEOUS_COURT = np.column_stack((CENTERED_COURT, np.ones(len(COURT))))
STRATA = (
    "all",
    "challenging",
    "low_angle",
    "oblique",
    "cropped",
    "side_on",
    "unclassified",
)


def capture_match(
    center: np.ndarray,
    forward: np.ndarray,
    hfov: float,
    capture_centers: np.ndarray,
    capture_forwards: np.ndarray,
    capture_hfovs: np.ndarray,
) -> tuple[float, int, np.ndarray]:
    """Minimize the largest normalized offset to one captured camera."""
    delta = capture_centers - center
    offsets = np.column_stack(
        (
            np.linalg.norm(delta[:, :2], axis=1),
            np.abs(delta[:, 2]),
            np.rad2deg(np.arccos(np.clip(capture_forwards @ forward, -1, 1))),
            np.abs(capture_hfovs - hfov),
        )
    )
    distances = (offsets / CAPTURE_LIMITS).max(axis=1)
    index = int(distances.argmin())
    return float(distances[index]), index, offsets[index]


def visible_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return ((points >= 0) & (points < np.array([width, height]))).all(axis=1)


def viewpoint_proxies(
    ground_truth: np.ndarray, width: int, height: int
) -> tuple[float, float | None] | None:
    """Estimate foreshortening and depth direction from annotated points."""
    matrix, _ = cv2.findHomography(CENTERED_COURT, ground_truth, method=0)
    if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        return None
    matrix = matrix / matrix[2, 2]
    homogeneous = HOMOGENEOUS_COURT @ matrix.T
    if np.any(homogeneous[:, 2] <= 0):
        return None
    projected = homogeneous[:, :2] / homogeneous[:, 2, None]
    if np.linalg.norm(projected - ground_truth, axis=1).mean() > 0.01 * math.hypot(
        width, height
    ):
        return None
    jacobian = matrix[:2, :2] - np.outer(matrix[:2, 2], matrix[2, :2])
    major, minor = np.linalg.svd(jacobian, compute_uv=False)
    if minor <= 0:
        return None
    elevation = math.degrees(math.asin(np.clip(minor / major, 0, 1)))
    depth = np.abs(matrix[2, :2])
    azimuth = (
        math.degrees(math.atan2(*depth))
        if np.ptp(homogeneous[:, 2]) >= MIN_DEPTH_SPAN
        else None
    )
    return elevation, azimuth


def annotate_view(sample_id: str, payload: Any) -> dict[str, Any]:
    width, height = int(payload["width"]), int(payload["height"])
    points = np.asarray(payload["keypoints"], dtype=np.float64)
    visible = visible_points(points, width, height)
    tags = ["all"]
    if not visible.all():
        tags.append("cropped")
    proxies = viewpoint_proxies(points, width, height)
    elevation, azimuth = (None, None) if proxies is None else proxies
    if elevation is not None and elevation <= LOW_ANGLE_DEG:
        tags.append("low_angle")
    if azimuth is not None and azimuth >= OBLIQUE_DEG:
        tags.append("oblique")
    if azimuth is not None and azimuth >= SIDE_ON_DEG:
        tags.append("side_on")
    if elevation is None or azimuth is None:
        tags.append("unclassified")
    if any(tag in tags for tag in ("low_angle", "oblique", "cropped")):
        tags.append("challenging")
    return {
        "sample_id": sample_id,
        "video_id": video_id(sample_id),
        "width": width,
        "height": height,
        "visible_kp": int(visible.sum()),
        "elevation_proxy_deg": elevation,
        "azimuth_proxy_deg": azimuth,
        "strata": tags,
    }


def stratum_counts(views: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = Counter(tag for view in views for tag in view["strata"])
    videos = {tag: set() for tag in STRATA}
    for view in views:
        for tag in view["strata"]:
            videos[tag].add(view["video_id"])
    return {tag: {"images": counts[tag], "videos": len(videos[tag])} for tag in STRATA}


def score_prediction(
    points: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
    ground_truth: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Score raw ordered peaks and test the fitted H against independent labels."""
    if points.shape != (14, 2) or ground_truth.shape != (14, 2):
        raise ValueError("Scoring requires ordered KP14 in original image pixels.")
    if valid.shape != (14,) or not np.isfinite(points).all() or not valid.all():
        raise ValueError("The threshold-free decoder must return all 14 finite peaks.")
    visible = visible_points(ground_truth, width, height)
    if visible.sum() < 4:
        raise ValueError("H accuracy requires at least four in-frame GT points.")
    raw_px = np.linalg.norm(points[visible] - ground_truth[visible], axis=1)
    diagonal = math.hypot(width, height)
    fit = estimate_confidence_homography(
        COURT,
        points,
        scores,
        reprojection_threshold_px=H_INLIER_RATIO * diagonal,
        min_score=H_MIN_SCORE,
    )
    h_error = None
    if fit.matrix is not None:
        errors = np.linalg.norm(fit.projected[visible] - ground_truth[visible], axis=1)
        h_error = float(errors.mean() / diagonal)
    return {
        "kp_errors_px512": (raw_px * (LONG_SIDE / max(width, height))).tolist(),
        "h_status": fit.status,
        "h_mean_error_diagonal": h_error,
        "h_at_1pct": h_error is not None and h_error <= H_ACCURACY_RATIO,
    }
