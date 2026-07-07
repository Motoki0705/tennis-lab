"""Homography-constrained court keypoint post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.geometry.homography import (
    court_template_xy,
    estimate_homography,
    project_points,
)


@dataclass(frozen=True)
class HomographyPostprocessResult:
    """Output of homography-constrained court keypoint refinement."""

    keypoints: NDArray[np.float32]
    visibility: NDArray[np.float32]
    diagnostics: dict[str, Any]


def refine_court_keypoints_with_homography(
    keypoints: NDArray[np.floating],
    scores: NDArray[np.floating],
    *,
    template_xy: NDArray[np.floating] | None = None,
    min_score: float = 0.3,
    ransac_reproj_threshold: float = 3.0,
    temporal_median_window: int = 0,
) -> HomographyPostprocessResult:
    """Refine a court keypoint sequence with a planar template homography.

    Frames with fewer than four high-score seed points, RANSAC failure, or
    degenerate refits are marked invisible instead of silently falling back.
    Successful frames replace every keypoint with the reprojected court
    template so downstream consumers receive geometrically consistent 14-KP
    predictions.
    """
    frame_keypoints = _as_keypoints_array(keypoints)
    frame_scores = _as_scores_array(scores, expected_shape=frame_keypoints.shape[:2])
    template = (
        court_template_xy(frame_keypoints.shape[1])
        if template_xy is None
        else _as_template_array(template_xy, expected_count=frame_keypoints.shape[1])
    )
    _validate_options(
        min_score=min_score,
        ransac_reproj_threshold=ransac_reproj_threshold,
        temporal_median_window=temporal_median_window,
    )

    refined = frame_keypoints.copy()
    visibility: NDArray[np.float32] = np.zeros(frame_keypoints.shape[:2], dtype=np.float32)
    frame_diagnostics: list[dict[str, Any]] = []

    for frame_index in range(frame_keypoints.shape[0]):
        refined_frame, diagnostics = _refine_frame(
            frame_keypoints[frame_index],
            frame_scores[frame_index],
            template,
            frame_index=frame_index,
            min_score=min_score,
            ransac_reproj_threshold=ransac_reproj_threshold,
        )
        frame_diagnostics.append(diagnostics)
        if refined_frame is None:
            continue
        refined[frame_index] = refined_frame
        visibility[frame_index] = 1.0

    if temporal_median_window > 0:
        refined = _apply_temporal_median_filter(
            refined,
            visibility,
            window=temporal_median_window,
        )

    diagnostics = {
        "config": {
            "min_score": float(min_score),
            "ransac_reproj_threshold": float(ransac_reproj_threshold),
            "temporal_median_window": int(temporal_median_window),
        },
        "num_frames": int(frame_keypoints.shape[0]),
        "num_success": int(sum(1 for item in frame_diagnostics if bool(item["success"]))),
        "frames": frame_diagnostics,
    }
    return HomographyPostprocessResult(
        keypoints=refined.astype(np.float32),
        visibility=visibility,
        diagnostics=diagnostics,
    )


def _refine_frame(
    keypoints: NDArray[np.float32],
    scores: NDArray[np.float32],
    template: NDArray[np.float32],
    *,
    frame_index: int,
    min_score: float,
    ransac_reproj_threshold: float,
) -> tuple[NDArray[np.float32] | None, dict[str, Any]]:
    seed_mask = scores >= float(min_score)
    seed_count = int(seed_mask.sum())
    diagnostics: dict[str, Any] = {
        "frame_index": int(frame_index),
        "success": False,
        "reason": None,
        "seed_count": seed_count,
        "inlier_count": 0,
        "residual_rms": None,
        "seed_mask": seed_mask.astype(int).tolist(),
        "inlier_mask": [0 for _ in range(template.shape[0])],
        "seed_residuals_px": [],
    }
    if seed_count < 4:
        diagnostics["reason"] = "insufficient_seed_points"
        return None, diagnostics

    seed_indices = np.flatnonzero(seed_mask)
    homography = estimate_homography(
        template[seed_mask],
        keypoints[seed_mask],
        method=int(cv2.RANSAC),
        ransac_reproj_threshold=ransac_reproj_threshold,
    )
    if homography is None:
        diagnostics["reason"] = "ransac_failed"
        return None, diagnostics

    projected_seed = project_points(template[seed_mask], homography)
    seed_residuals = _point_distances(projected_seed, keypoints[seed_mask])
    inlier_seed_mask = seed_residuals <= float(ransac_reproj_threshold)
    inlier_count = int(inlier_seed_mask.sum())
    full_inlier_mask = np.zeros(template.shape[0], dtype=bool)
    full_inlier_mask[seed_indices[inlier_seed_mask]] = True
    diagnostics["seed_residuals_px"] = _float_list(seed_residuals)
    diagnostics["inlier_count"] = inlier_count
    diagnostics["inlier_mask"] = full_inlier_mask.astype(int).tolist()

    if inlier_count < 4:
        diagnostics["reason"] = "insufficient_inliers"
        return None, diagnostics

    refit_homography = estimate_homography(
        template[full_inlier_mask],
        keypoints[full_inlier_mask],
        method=0,
        ransac_reproj_threshold=ransac_reproj_threshold,
    )
    if refit_homography is None:
        diagnostics["reason"] = "refit_failed"
        return None, diagnostics

    try:
        refined = project_points(template, refit_homography)
    except ValueError:
        diagnostics["reason"] = "degenerate_homography"
        return None, diagnostics
    if not _projected_template_is_valid(refined):
        diagnostics["reason"] = "degenerate_homography"
        return None, diagnostics

    refit_inlier_projection = project_points(template[full_inlier_mask], refit_homography)
    refit_residuals = _point_distances(
        refit_inlier_projection,
        keypoints[full_inlier_mask],
    )
    residual_rms = float(np.sqrt(np.mean(refit_residuals * refit_residuals)))
    diagnostics["success"] = True
    diagnostics["residual_rms"] = residual_rms
    return refined, diagnostics


def _apply_temporal_median_filter(
    keypoints: NDArray[np.float32],
    visibility: NDArray[np.float32],
    *,
    window: int,
) -> NDArray[np.float32]:
    filtered = keypoints.copy()
    radius = window // 2
    num_frames, num_keypoints = keypoints.shape[:2]
    for frame_index in range(num_frames):
        start = max(0, frame_index - radius)
        stop = min(num_frames, frame_index + radius + 1)
        for keypoint_index in range(num_keypoints):
            if visibility[frame_index, keypoint_index] <= 0.0:
                continue
            valid = visibility[start:stop, keypoint_index] > 0.0
            if not bool(valid.any()):
                continue
            values = keypoints[start:stop, keypoint_index][valid]
            filtered[frame_index, keypoint_index] = np.median(values, axis=0)
    return filtered


def _point_distances(
    lhs: NDArray[np.float32],
    rhs: NDArray[np.float32],
) -> NDArray[np.float32]:
    distances = np.linalg.norm(lhs - rhs, axis=1)
    return cast(NDArray[np.float32], np.asarray(distances, dtype=np.float32))


def _projected_template_is_valid(points: NDArray[np.float32]) -> bool:
    if points.shape[0] < 4 or not np.isfinite(points).all():
        return False
    x_span = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    y_span = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    return x_span > 1.0e-3 and y_span > 1.0e-3


def _float_list(values: NDArray[np.float32]) -> list[float]:
    return [float(value) for value in values.tolist()]


def _as_keypoints_array(keypoints: NDArray[np.floating]) -> NDArray[np.float32]:
    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"keypoints must have shape (T, K, 2), got {array.shape}.")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"keypoints must contain at least one frame and KP, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("keypoints must contain only finite coordinates.")
    return cast(NDArray[np.float32], array)


def _as_scores_array(
    scores: NDArray[np.floating],
    *,
    expected_shape: tuple[int, int],
) -> NDArray[np.float32]:
    array = np.asarray(scores, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"scores must have shape {expected_shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("scores must contain only finite values.")
    return cast(NDArray[np.float32], array)


def _as_template_array(
    template_xy: NDArray[np.floating],
    *,
    expected_count: int,
) -> NDArray[np.float32]:
    template = np.asarray(template_xy, dtype=np.float32)
    if template.shape != (expected_count, 2):
        raise ValueError(f"template_xy must have shape ({expected_count}, 2), got {template.shape}.")
    if not np.isfinite(template).all():
        raise ValueError("template_xy must contain only finite coordinates.")
    return cast(NDArray[np.float32], template)


def _validate_options(
    *,
    min_score: float,
    ransac_reproj_threshold: float,
    temporal_median_window: int,
) -> None:
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score}.")
    if ransac_reproj_threshold <= 0:
        raise ValueError(
            "ransac_reproj_threshold must be positive, "
            f"got {ransac_reproj_threshold}."
        )
    if temporal_median_window < 0 or (
        temporal_median_window != 0 and temporal_median_window % 2 == 0
    ):
        raise ValueError(
            "temporal_median_window must be 0 or a positive odd integer, "
            f"got {temporal_median_window}."
        )
