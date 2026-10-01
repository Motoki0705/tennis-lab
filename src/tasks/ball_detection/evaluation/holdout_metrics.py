"""Source-pixel holdout metrics with explicit missing and uncertain references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    POINT_KIND_NAMES,
    BallFrameStore,
    ClipRecord,
)
from src.tasks.ball_detection.evaluation.holdout_inference import FramePredictions


@dataclass(frozen=True)
class HoldoutReferences:
    row: NDArray[np.int64]
    clip_id: NDArray[np.str_]
    camera: NDArray[np.str_]
    frame_index: NDArray[np.int64]
    annotated: NDArray[np.bool_]
    point_kind: NDArray[np.str_]
    occluded: NDArray[np.bool_]
    uv: NDArray[np.float32]
    wrist_distance: NDArray[np.float32]

    def __post_init__(self) -> None:
        n = len(self.row)
        columns = (self.row, self.clip_id, self.camera, self.frame_index,
                   self.annotated, self.point_kind, self.occluded, self.wrist_distance)
        if any(x.shape != (n,) for x in columns) or self.uv.shape != (n, 2):
            raise ValueError("Reference arrays must share one frame timeline")
        if len(np.unique(self.row)) != n:
            raise ValueError("Each store frame must be evaluated exactly once")
        located = np.isin(self.point_kind, ["observed", "interpolated", "occlusion_estimated"])
        if not np.isfinite(self.uv[located]).all() or not np.isnan(self.uv[~located]).all():
            raise ValueError("Only located references may carry coordinates")
        if np.isinf(self.wrist_distance).any() or (self.wrist_distance < 0).any():
            raise ValueError("Wrist distances must be nonnegative or NaN (unknown)")


def clip_references(store: BallFrameStore, clip: ClipRecord) -> HoldoutReferences:
    """Meiji single-ball references; reject multi-instance data, never choose a GT."""
    rows = store.clip_rows(clip)
    kinds: list[str] = []
    uv = np.full((clip.frame_count, 2), np.nan, np.float32)
    occluded = np.zeros(clip.frame_count, bool)
    for i, row in enumerate(rows):
        instances = store.instances_of(int(row))
        if len(instances.xy) > 1:
            raise ValueError(f"Single-ball holdout has multiple references: {clip.clip_id}:{i}")
        kind = "no_instance"
        if len(instances.xy):
            kind = POINT_KIND_NAMES[int(instances.point_kind[0])]
            uv[i] = instances.xy[0] / clip.scale
            occluded[i] = instances.occluded[0]
        kinds.append(kind)
    if clip.camera_id is None:
        raise ValueError("Stratified holdout requires a camera ID")
    return HoldoutReferences(
        rows, np.full(len(rows), clip.clip_id), np.full(len(rows), clip.camera_id),
        np.arange(len(rows), dtype=np.int64), store.frames["annotated"][rows],
        np.asarray(kinds), occluded, uv, np.full(len(rows), np.nan, np.float32),
    )


def wrist_distances(
    ball_uv: NDArray[np.float32], wrist_uv: NDArray[np.float32],
    wrist_valid: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Nearest valid source-pixel wrist; missing wrists/ball stay unknown."""
    if wrist_uv.ndim != 3 or wrist_uv.shape[::2] != (len(ball_uv), 2):
        raise ValueError("Wrist coordinates must be (frames, wrists, 2)")
    if wrist_valid.shape != wrist_uv.shape[:-1] or wrist_valid.dtype != np.bool_:
        raise ValueError("Wrist validity must match its coordinates")
    if not np.isfinite(wrist_uv[wrist_valid]).all():
        raise ValueError("Valid wrists must have finite coordinates")
    result: NDArray[np.float32] = np.full(len(ball_uv), np.nan, np.float32)
    if wrist_uv.shape[1] == 0:
        return result
    distances = np.linalg.norm(wrist_uv - ball_uv[:, None], axis=-1)
    nearest = np.min(np.where(wrist_valid, distances, np.inf), axis=1)
    known = np.isfinite(ball_uv).all(axis=1) & np.isfinite(nearest)
    result[known] = nearest[known]
    return result


def summarize_holdout(
    reference: HoldoutReferences, prediction: FramePredictions, *,
    score_threshold: float, distance_px: float, near_wrist_px: float,
) -> dict[str, dict[str, Any]]:
    """Primary rows use observed labels; estimated locations are separate rows.

    Accepted p95 includes *every* accepted coordinate reference, even large
    errors. Raw-argmax p95 additionally includes low-confidence predictions.
    Unknown GT is never a negative. Empty denominators/quantiles are null.
    """
    if len(reference.row) != len(prediction.score):
        raise ValueError("Prediction/reference frame counts disagree")
    if (not np.isfinite([score_threshold, distance_px, near_wrist_px]).all()
            or not 0 <= score_threshold <= 1 or min(distance_px, near_wrist_px) <= 0):
        raise ValueError("Invalid holdout thresholds")
    located = reference.annotated & np.isfinite(reference.uv).all(axis=1)
    observed = reference.annotated & (reference.point_kind == "observed")
    detected = prediction.score >= score_threshold
    errors = np.linalg.norm(prediction.uv - reference.uv, axis=1)
    candidate_errors = np.linalg.norm(prediction.candidate_uv - reference.uv[:, None], axis=-1)
    candidate_hit = np.any(prediction.candidate_valid & (candidate_errors <= distance_px), axis=1)
    wrist_kind = np.where(
        np.isnan(reference.wrist_distance), "unknown",
        np.where(reference.wrist_distance <= near_wrist_px, "near_wrist", "flight"),
    )
    groups = {"overall_observed": observed}
    for camera in np.unique(reference.camera):
        groups[f"camera/{camera}"] = observed & (reference.camera == camera)
        for wrist in ("near_wrist", "flight", "unknown"):
            groups[f"camera_wrist/{camera}/{wrist}"] = observed & (reference.camera == camera) & (wrist_kind == wrist)
    for kind in (*POINT_KIND_NAMES.values(), "no_instance"):
        groups[f"point_kind/{kind}"] = reference.annotated & (reference.point_kind == kind)
    groups["visibility/not_occluded"] = located & ~reference.occluded
    groups["visibility/occluded"] = located & reference.occluded
    groups["visibility/unlocated_or_unreviewed"] = ~located
    for wrist in ("near_wrist", "flight", "unknown"):
        groups[f"wrist/{wrist}"] = observed & (wrist_kind == wrist)
    result: dict[str, dict[str, Any]] = {}
    for name, mask in groups.items():
        scored = mask & located
        accepted = scored & detected
        hits = accepted & (errors <= distance_px)
        count, accepted_count, hit_count = int(scored.sum()), int(accepted.sum()), int(hits.sum())
        result[name] = {
            "frames": int(mask.sum()), "reference_frames": count,
            "without_reference_frames": int((mask & ~located).sum()),
            "detected_without_reference_frames": int((mask & ~located & detected).sum()),
            "detected_reference_frames": accepted_count,
            "missing_reference_frames": count - accepted_count,
            "wrong_reference_frames": accepted_count - hit_count,
            "matched_reference_frames": hit_count,
            "recall": hit_count / count if count else None,
            "topk_recall_unthresholded": int((scored & candidate_hit).sum()) / count if count else None,
            "accepted_p95_px": float(np.quantile(errors[accepted], 0.95)) if accepted_count else None,
            "raw_argmax_p95_px": float(np.quantile(errors[scored], 0.95)) if count else None,
        }
    return result
