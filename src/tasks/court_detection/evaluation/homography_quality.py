"""Pure homography fitting and projected-court quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.contracts import (
    HomographyEvaluationCriteria,
)
from src.tasks.court_detection.geometry import court_template_xy, project_points

OUTER_COURT_INDICES: NDArray[np.int32] = np.asarray([0, 1, 3, 2], dtype=np.int32)


@dataclass(frozen=True)
class HomographyQualityResult:
    """Homography fit plus all geometry-only acceptance diagnostics."""

    homography: NDArray[np.float32] | None
    projected_keypoints_normalized: NDArray[np.float32] | None
    inlier_mask: NDArray[np.bool_]
    residuals_normalized: NDArray[np.float32]
    metrics: dict[str, float | int]
    rejection_reasons: tuple[str, ...]

    @property
    def geometry_valid(self) -> bool:
        """Whether a finite, non-degenerate homography was produced."""
        return (
            self.homography is not None
            and self.projected_keypoints_normalized is not None
        )


def evaluate_homography_quality(
    keypoints_xy: NDArray[np.floating],
    *,
    image_width: int,
    image_height: int,
    criteria: HomographyEvaluationCriteria,
) -> HomographyQualityResult:
    """Fit and validate the canonical 14-point court against one annotation."""
    keypoints = _as_keypoints(keypoints_xy)
    if image_width <= 1 or image_height <= 1:
        raise ValueError(
            "Image dimensions must both be greater than one, "
            f"got width={image_width}, height={image_height}."
        )
    normalization = np.asarray(
        [image_width - 1, image_height - 1],
        dtype=np.float32,
    )
    normalized = keypoints / normalization
    template = court_template_xy()
    empty_mask: NDArray[np.bool_] = np.zeros(template.shape[0], dtype=bool)
    empty_residuals: NDArray[np.float32] = np.full(
        template.shape[0],
        np.inf,
        dtype=np.float32,
    )

    homography_raw, status = _find_homography(
        template,
        normalized,
        method=int(cv2.RANSAC),
        ransac_reproj_threshold=float(criteria.ransac_reproj_threshold_normalized),
    )
    if homography_raw is None or status is None:
        return HomographyQualityResult(
            homography=None,
            projected_keypoints_normalized=None,
            inlier_mask=empty_mask,
            residuals_normalized=empty_residuals,
            metrics={"inlier_count": 0},
            rejection_reasons=("ransac_failed",),
        )

    initial = np.asarray(homography_raw, dtype=np.float32)
    try:
        initial_projection = project_points(template, initial)
    except ValueError:
        return HomographyQualityResult(
            homography=None,
            projected_keypoints_normalized=None,
            inlier_mask=empty_mask,
            residuals_normalized=empty_residuals,
            metrics={"inlier_count": 0},
            rejection_reasons=("degenerate_homography",),
        )

    residuals = _distances(initial_projection, normalized)
    inlier_mask = residuals <= float(criteria.ransac_reproj_threshold_normalized)
    inlier_count = int(inlier_mask.sum())
    metrics: dict[str, float | int] = {"inlier_count": inlier_count}
    reasons: list[str] = []
    if inlier_count < criteria.min_inliers:
        reasons.append("insufficient_inliers")

    x_coverage, y_coverage = _template_coverage(template, inlier_mask)
    metrics["template_x_span_ratio"] = x_coverage
    metrics["template_y_span_ratio"] = y_coverage
    if (
        x_coverage < criteria.min_template_x_span_ratio
        or y_coverage < criteria.min_template_y_span_ratio
    ):
        reasons.append("degenerate_inlier_coverage")
    if reasons:
        return HomographyQualityResult(
            homography=None,
            projected_keypoints_normalized=None,
            inlier_mask=inlier_mask,
            residuals_normalized=residuals,
            metrics=metrics,
            rejection_reasons=tuple(reasons),
        )

    refit_raw, _ = _find_homography(
        template[inlier_mask],
        normalized[inlier_mask],
        method=0,
        ransac_reproj_threshold=criteria.ransac_reproj_threshold_normalized,
    )
    if refit_raw is None:
        return HomographyQualityResult(
            homography=None,
            projected_keypoints_normalized=None,
            inlier_mask=inlier_mask,
            residuals_normalized=residuals,
            metrics=metrics,
            rejection_reasons=("refit_failed",),
        )
    refit = np.asarray(refit_raw, dtype=np.float32)
    try:
        projected = project_points(template, refit)
    except ValueError:
        return HomographyQualityResult(
            homography=None,
            projected_keypoints_normalized=None,
            inlier_mask=inlier_mask,
            residuals_normalized=residuals,
            metrics=metrics,
            rejection_reasons=("degenerate_homography",),
        )
    if not _projected_court_is_valid(projected):
        return HomographyQualityResult(
            homography=None,
            projected_keypoints_normalized=None,
            inlier_mask=inlier_mask,
            residuals_normalized=residuals,
            metrics=metrics,
            rejection_reasons=("degenerate_homography",),
        )

    refit_residuals = _distances(projected, normalized)
    inlier_rms = float(
        np.sqrt(np.mean(np.square(refit_residuals[inlier_mask]), dtype=np.float64))
    )
    all_rms = float(np.sqrt(np.mean(np.square(refit_residuals), dtype=np.float64)))
    metrics.update(
        {
            "inlier_residual_rms_normalized": inlier_rms,
            "all_residual_rms_normalized": all_rms,
            **projected_court_metrics(projected),
        }
    )
    if inlier_rms > criteria.max_inlier_rms_normalized:
        reasons.append("high_reprojection_error")
    visible_fraction = float(metrics["visible_fraction"])
    if visible_fraction < criteria.min_visible_fraction:
        reasons.append("court_not_fully_visible")
    area_ratio = float(metrics["court_area_ratio"])
    if area_ratio < criteria.min_court_area_ratio:
        reasons.append("court_too_small")
    if area_ratio > criteria.max_court_area_ratio:
        reasons.append("court_too_large")
    if (
        criteria.require_ground_view
        and float(metrics["opposite_edge_ratio"]) > criteria.max_opposite_edge_ratio
    ):
        reasons.append("insufficient_ground_perspective")

    return HomographyQualityResult(
        homography=refit,
        projected_keypoints_normalized=projected,
        inlier_mask=inlier_mask,
        residuals_normalized=refit_residuals,
        metrics=metrics,
        rejection_reasons=tuple(reasons),
    )


def projected_court_metrics(
    projected_keypoints_normalized: NDArray[np.floating],
) -> dict[str, float]:
    """Measure court occupancy, visibility, and projective distortion."""
    points = np.asarray(projected_keypoints_normalized, dtype=np.float32)
    if points.shape != (14, 2) or not np.isfinite(points).all():
        raise ValueError(
            f"Expected finite projected keypoints with shape (14, 2), got {points.shape}."
        )
    outer = points[OUTER_COURT_INDICES]
    area = abs(float(cv2.contourArea(outer)))
    image_box = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    visible_area, _ = cv2.intersectConvexConvex(cv2.convexHull(outer), image_box)
    visible_fraction = max(float(visible_area), 0.0) / area if area > 0.0 else 0.0

    far_width = float(np.linalg.norm(points[0] - points[1]))
    near_width = float(np.linalg.norm(points[2] - points[3]))
    left_length = float(np.linalg.norm(points[0] - points[2]))
    right_length = float(np.linalg.norm(points[1] - points[3]))
    baseline_ratio = _smaller_to_larger_ratio(far_width, near_width)
    sideline_ratio = _smaller_to_larger_ratio(left_length, right_length)
    return {
        "court_area_ratio": area,
        "visible_fraction": min(visible_fraction, 1.0),
        "baseline_width_ratio": baseline_ratio,
        "sideline_length_ratio": sideline_ratio,
        "opposite_edge_ratio": min(baseline_ratio, sideline_ratio),
    }


def _as_keypoints(keypoints_xy: NDArray[np.floating]) -> NDArray[np.float32]:
    keypoints = np.asarray(keypoints_xy, dtype=np.float32)
    if keypoints.shape != (14, 2):
        raise ValueError(f"keypoints must have shape (14, 2), got {keypoints.shape}.")
    if not np.isfinite(keypoints).all():
        raise ValueError("keypoints must contain only finite coordinates.")
    return keypoints


def _distances(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
) -> NDArray[np.float32]:
    values = np.linalg.norm(first - second, axis=1)
    return cast(NDArray[np.float32], np.asarray(values, dtype=np.float32))


def _template_coverage(
    template: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> tuple[float, float]:
    if not bool(mask.any()):
        return 0.0, 0.0
    full_span = np.ptp(template, axis=0)
    inlier_span = np.ptp(template[mask], axis=0)
    return (
        float(inlier_span[0] / full_span[0]),
        float(inlier_span[1] / full_span[1]),
    )


def _projected_court_is_valid(points: NDArray[np.float32]) -> bool:
    outer = points[OUTER_COURT_INDICES]
    if not np.isfinite(points).all() or not cv2.isContourConvex(outer):
        return False
    return abs(float(cv2.contourArea(outer))) > 1.0e-8


def _smaller_to_larger_ratio(first: float, second: float) -> float:
    return min(first, second) / max(first, second, 1.0e-8)


def _find_homography(
    source: NDArray[np.float32],
    destination: NDArray[np.float32],
    *,
    method: int,
    ransac_reproj_threshold: float,
) -> tuple[NDArray[np.float32] | None, NDArray[np.uint8] | None]:
    raw_result: Any = cv2.findHomography(
        source,
        destination,
        method,
        float(ransac_reproj_threshold),
    )
    homography_raw, status_raw = raw_result
    homography = (
        np.asarray(homography_raw, dtype=np.float32)
        if homography_raw is not None
        else None
    )
    status = np.asarray(status_raw, dtype=np.uint8) if status_raw is not None else None
    return homography, status
