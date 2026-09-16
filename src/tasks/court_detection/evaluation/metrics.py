"""Strict cross-model keypoint, homography, and court-line metrics.

Every metric states its denominator.  Ground-truth-visible keypoints form the
keypoint denominator, and a missing prediction is an incorrect prediction
rather than a dropped sample; the valid-pair error distributions keep their
own explicit pair count next to the mean.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.alignment import (
    COURT_LINE_SEGMENTS,
    canonical_keypoint_names,
    doubles_polygon_template,
    fit_template_homography,
    points_inside_image,
    polygon_iou,
    project_points,
    sample_segment_points,
    symmetric_line_reprojection_error_px,
)
from src.tasks.court_detection.evaluation.contracts import (
    KEYPOINT_COUNT,
    LoadedSample,
    ModelPrediction,
    SampleRef,
)
from src.tasks.court_detection.evaluation.settings import BenchmarkQualitySettings

REASON_NO_VISIBLE_GT = "no_visible_ground_truth_keypoints"
REASON_GT_HOMOGRAPHY_FAILED = "ground_truth_homography_failed"
REASON_NO_PAIR = "no_valid_prediction_court_homography"
REASON_NO_IN_FRAME_LINE_SAMPLES = "no_in_frame_line_samples"


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    """Everything one sample contributes to the aggregate report."""

    ref: SampleRef
    model: str
    strata: Mapping[str, str]
    gt_visible: NDArray[np.bool_]
    pred_valid: NDArray[np.bool_]
    errors_px: NDArray[np.float64]  # NaN unless the pair exists
    gt_homography: NDArray[np.float64] | None
    pred_homography: NDArray[np.float64] | None
    pred_homography_reason: str | None
    pred_homography_inliers: int
    line_error_px: float | None
    line_error_in_frame_px: float | None
    line_samples_in_frame: int
    line_samples_total: int
    polygon_iou: float | None
    undefined_reasons: tuple[str, ...]

    @property
    def pair_errors_px(self) -> NDArray[np.float64]:
        values = self.errors_px[np.isfinite(self.errors_px)]
        return cast("NDArray[np.float64]", values)

    @property
    def median_error_px(self) -> float | None:
        values = self.pair_errors_px
        if values.size == 0:
            return None
        return float(np.median(values))

    @property
    def gt_visible_count(self) -> int:
        return int(self.gt_visible.sum())


def evaluate_sample(
    sample: LoadedSample,
    prediction: ModelPrediction,
    *,
    quality: BenchmarkQualitySettings,
) -> SampleEvaluation:
    """Score one model prediction against one loaded ground-truth sample."""
    diagonal = sample.ref.diagonal_px
    threshold = quality.ransac_threshold_fraction * diagonal
    image_points = prediction.keypoints.keypoints_xy
    pred_valid = np.asarray(prediction.keypoints.valid, dtype=bool)

    gt_visible = np.asarray(sample.gt_visible, dtype=bool)
    if gt_visible.shape != (KEYPOINT_COUNT,) or pred_valid.shape != (KEYPOINT_COUNT,):
        raise ValueError("Keypoint masks must have shape (14,).")

    errors: NDArray[np.float64] = np.full(KEYPOINT_COUNT, np.nan, dtype=np.float64)
    pairs = gt_visible & pred_valid & np.isfinite(image_points).all(axis=1)
    if pairs.any():
        errors[pairs] = np.linalg.norm(
            sample.gt_keypoints_xy[pairs] - image_points[pairs], axis=1
        )

    lines = sample_segment_points(
        COURT_LINE_SEGMENTS,
        sample.template_xy,
        per_segment=quality.line_samples_per_segment,
    )
    gt_fit = fit_template_homography(
        sample.template_xy,
        sample.gt_keypoints_xy,
        gt_visible,
        ransac_threshold_px=threshold,
    )
    pred_fit = fit_template_homography(
        sample.template_xy,
        image_points,
        pred_valid,
        ransac_threshold_px=threshold,
    )

    line_error: float | None = None
    line_error_in_frame: float | None = None
    in_frame_count = 0
    iou: float | None = None
    reasons: list[str] = []
    if not gt_visible.any():
        reasons.append(REASON_NO_VISIBLE_GT)
    if gt_fit.homography is None:
        reasons.append(REASON_GT_HOMOGRAPHY_FAILED)
    if pred_fit.homography is None:
        if pred_fit.reason is not None:
            reasons.append(pred_fit.reason)
        reasons.append(REASON_NO_PAIR)
    else:
        polygon = doubles_polygon_template(sample.template_xy)
        if gt_fit.homography is not None:
            line_error = symmetric_line_reprojection_error_px(
                gt_fit.homography, pred_fit.homography, lines
            )
            reference_lines = project_points(lines, gt_fit.homography)
            inside = points_inside_image(
                reference_lines, sample.ref.width, sample.ref.height
            )
            in_frame_count = int(inside.sum())
            if in_frame_count > 0:
                line_error_in_frame = symmetric_line_reprojection_error_px(
                    gt_fit.homography,
                    pred_fit.homography,
                    lines,
                    include=inside,
                )
            else:
                reasons.append(REASON_NO_IN_FRAME_LINE_SAMPLES)
            iou, polygon_reason = polygon_iou(
                project_points(polygon, gt_fit.homography),
                project_points(polygon, pred_fit.homography),
                width=sample.ref.width,
                height=sample.ref.height,
            )
            if polygon_reason is not None:
                reasons.append(polygon_reason)
    return SampleEvaluation(
        ref=sample.ref,
        model=prediction.model,
        strata=sample.strata,
        gt_visible=gt_visible,
        pred_valid=pred_valid,
        errors_px=errors,
        gt_homography=gt_fit.homography,
        pred_homography=pred_fit.homography,
        pred_homography_reason=pred_fit.reason,
        pred_homography_inliers=pred_fit.inlier_count,
        line_error_px=line_error,
        line_error_in_frame_px=line_error_in_frame,
        line_samples_in_frame=in_frame_count,
        line_samples_total=int(lines.shape[0]),
        polygon_iou=iou,
        undefined_reasons=tuple(sorted(set(reasons))),
    )


def _stats(values: Sequence[float] | NDArray[np.floating]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q90": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
    }


def _pck_curve(
    evaluations: Sequence[SampleEvaluation],
    *,
    fractions: tuple[float, ...],
) -> dict[str, float | None]:
    denominator = int(sum(item.gt_visible_count for item in evaluations))
    curve: dict[str, float | None] = {}
    for fraction in fractions:
        if denominator == 0:
            curve[f"{fraction}"] = None
            continue
        correct = 0
        for item in evaluations:
            threshold = fraction * item.ref.diagonal_px
            hits = np.isfinite(item.errors_px) & (item.errors_px <= threshold)
            correct += int(hits.sum())
        curve[f"{fraction}"] = float(correct / denominator)
    return curve


def _keypoint_summary(
    evaluations: Sequence[SampleEvaluation],
    *,
    quality: BenchmarkQualitySettings,
) -> dict[str, object]:
    gt_total = int(sum(item.gt_visible_count for item in evaluations))
    predicted_valid = int(
        sum(int((item.gt_visible & item.pred_valid).sum()) for item in evaluations)
    )
    completeness = float(predicted_valid / gt_total) if gt_total else None
    all_errors = (
        np.concatenate([item.pair_errors_px for item in evaluations])
        if evaluations
        else np.empty(0, dtype=np.float64)
    )
    normalized = np.asarray(
        [
            value / item.ref.diagonal_px
            for item in evaluations
            for value in item.pair_errors_px
        ],
        dtype=np.float64,
    )
    return {
        "sample_count": len(evaluations),
        "ground_truth_visible_keypoints": gt_total,
        "predicted_valid_keypoints_among_visible": predicted_valid,
        "completeness": completeness,
        "pck": _pck_curve(evaluations, fractions=quality.pck_fractions),
        "pair_error_px": _stats(all_errors),
        "pair_error_diagonal": _stats(normalized),
    }


def _alignment_summary(evaluations: Sequence[SampleEvaluation]) -> dict[str, object]:
    sample_count = len(evaluations)
    successes = sum(1 for item in evaluations if item.pred_homography is not None)
    failure_reasons = Counter(
        item.pred_homography_reason
        for item in evaluations
        if item.pred_homography is None and item.pred_homography_reason is not None
    )
    undefined = Counter(
        reason for item in evaluations for reason in item.undefined_reasons
    )
    line_errors = np.asarray(
        [item.line_error_px for item in evaluations if item.line_error_px is not None],
        dtype=np.float64,
    )
    line_errors_normalized = np.asarray(
        [
            item.line_error_px / item.ref.diagonal_px
            for item in evaluations
            if item.line_error_px is not None
        ],
        dtype=np.float64,
    )
    in_frame_errors = np.asarray(
        [
            item.line_error_in_frame_px
            for item in evaluations
            if item.line_error_in_frame_px is not None
        ],
        dtype=np.float64,
    )
    in_frame_normalized = np.asarray(
        [
            item.line_error_in_frame_px / item.ref.diagonal_px
            for item in evaluations
            if item.line_error_in_frame_px is not None
        ],
        dtype=np.float64,
    )
    total_samples = sum(item.line_samples_total for item in evaluations)
    in_frame_samples = sum(item.line_samples_in_frame for item in evaluations)
    ious = np.asarray(
        [item.polygon_iou for item in evaluations if item.polygon_iou is not None],
        dtype=np.float64,
    )
    return {
        "sample_count": sample_count,
        "ground_truth_homography_success": sum(
            1 for item in evaluations if item.gt_homography is not None
        ),
        "predicted_homography_success": successes,
        "predicted_homography_success_rate": (
            float(successes / sample_count) if sample_count else None
        ),
        "predicted_homography_failure_reasons": dict(sorted(failure_reasons.items())),
        # Inlier counts are only meaningful for fits that succeeded; failed fits
        # are counted by the success rate and failure reasons instead of being
        # averaged in as zeros.
        "predicted_homography_inliers_on_success": _stats(
            [
                float(item.pred_homography_inliers)
                for item in evaluations
                if item.pred_homography is not None
            ]
        ),
        "line_reprojection_px": _stats(line_errors),
        "line_reprojection_diagonal": _stats(line_errors_normalized),
        "line_reprojection_in_frame_px": _stats(in_frame_errors),
        "line_reprojection_in_frame_diagonal": _stats(in_frame_normalized),
        "line_samples_in_frame_fraction": (
            float(in_frame_samples / total_samples) if total_samples else None
        ),
        "line_samples_in_frame": in_frame_samples,
        "line_samples_total": total_samples,
        "doubles_polygon_iou": _stats(ious),
        "undefined_reasons": dict(sorted(undefined.items())),
    }


def _per_channel_summary(
    evaluations: Sequence[SampleEvaluation],
    *,
    quality: BenchmarkQualitySettings,
) -> list[dict[str, object]]:
    names = canonical_keypoint_names()
    diagonals = np.asarray(
        [item.ref.diagonal_px for item in evaluations], dtype=np.float64
    )
    rows: list[dict[str, object]] = []
    for index in range(KEYPOINT_COUNT):
        gt_visible = np.asarray(
            [item.gt_visible[index] for item in evaluations], dtype=bool
        )
        pred_valid = np.asarray(
            [item.pred_valid[index] for item in evaluations], dtype=bool
        )
        errors = np.asarray(
            [item.errors_px[index] for item in evaluations], dtype=np.float64
        )
        denominator = int(gt_visible.sum())
        pck: dict[str, float | None] = {}
        for fraction in quality.pck_fractions:
            if denominator == 0:
                pck[f"{fraction}"] = None
                continue
            hits = (
                gt_visible
                & pred_valid
                & np.isfinite(errors)
                & (errors <= fraction * diagonals)
            )
            pck[f"{fraction}"] = float(int(hits.sum()) / denominator)
        rows.append(
            {
                "index": index,
                "name": names[index],
                "ground_truth_visible": denominator,
                "predicted_valid_among_visible": int((gt_visible & pred_valid).sum()),
                "pck": pck,
                "pair_error_px": _stats(errors),
            }
        )
    return rows


def _strata_summary(
    evaluations: Sequence[SampleEvaluation],
    *,
    quality: BenchmarkQualitySettings,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, list[SampleEvaluation]]] = {
        axis: {} for axis in quality.strata_axes
    }
    for item in evaluations:
        for axis in quality.strata_axes:
            value = item.strata.get(axis)
            if value is None:
                continue
            grouped[axis].setdefault(value, []).append(item)
    result: dict[str, dict[str, object]] = {}
    for axis, buckets in grouped.items():
        result[axis] = {
            value: {
                **_keypoint_summary(items, quality=quality),
                "alignment": _alignment_summary(items),
            }
            for value, items in sorted(buckets.items())
        }
    return result


def aggregate_evaluations(
    evaluations: Sequence[SampleEvaluation],
    *,
    quality: BenchmarkQualitySettings,
) -> dict[str, object]:
    """Aggregate one model's evaluations into the published metric bundle."""
    if not evaluations:
        raise ValueError("Aggregation requires at least one sample evaluation.")
    models = {item.model for item in evaluations}
    domains = {item.ref.domain for item in evaluations}
    if len(models) != 1 or len(domains) != 1:
        raise ValueError("One aggregate must contain exactly one model and one domain.")
    return {
        "model": evaluations[0].model,
        "domain": evaluations[0].ref.domain,
        "keypoints": _keypoint_summary(evaluations, quality=quality),
        "alignment": _alignment_summary(evaluations),
        "per_channel": _per_channel_summary(evaluations, quality=quality),
        "strata": _strata_summary(evaluations, quality=quality),
    }


__all__ = [
    "REASON_GT_HOMOGRAPHY_FAILED",
    "REASON_NO_PAIR",
    "REASON_NO_VISIBLE_GT",
    "SampleEvaluation",
    "aggregate_evaluations",
    "evaluate_sample",
]
