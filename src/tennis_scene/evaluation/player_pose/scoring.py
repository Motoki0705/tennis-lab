"""Partial-label detection and COCO17 pose scoring on native-pixel inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.bbox import match_iou, pairwise_iou
from src.utils.schema.player import COCO_KP_NAMES, NUM_HUMAN_KP

IOU_CUTOFFS = (0.50, 0.75)
NO_PRECISION_REASON = (
    "Only selected players are labelled. Unmatched predictions have unknown status."
)
# COCOeval's COCO17 sigmas, in COCO_KP_NAMES order.
COCO_SIGMAS = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
    dtype=np.float64,
)


class FrameKey(NamedTuple):
    clip_id: str
    camera_id: str
    source_frame: int


def _check_boxes(boxes: NDArray[np.floating], count: int) -> None:
    if (
        boxes.shape != (count, 4)
        or not np.isfinite(boxes).all()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError("Boxes must be finite positive-area xyxy rectangles")


@dataclass(frozen=True)
class ReferenceFrame:
    """Only eligible player boxes, with scoreable COCO17 joints marked explicitly."""

    player_ids: NDArray[np.integer]
    boxes_xyxy: NDArray[np.floating]
    keypoints_xy: NDArray[np.floating]
    joint_valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        count = len(self.player_ids)
        _check_boxes(self.boxes_xyxy, count)
        if (
            self.player_ids.shape != (count,)
            or not np.issubdtype(self.player_ids.dtype, np.integer)
            or len(np.unique(self.player_ids)) != count
            or self.keypoints_xy.shape != (count, NUM_HUMAN_KP, 2)
            or self.joint_valid.shape != (count, NUM_HUMAN_KP)
            or self.joint_valid.dtype != np.bool_
            or not np.isfinite(self.keypoints_xy[self.joint_valid]).all()
        ):
            raise ValueError(
                "Expected unique player IDs and COCO17 pixels with boolean joint validity"
            )


@dataclass(frozen=True)
class Predictions:
    """One image's candidate boxes and aligned optional Pose2DResult arrays."""

    boxes_xyxy: NDArray[np.floating]
    scores: NDArray[np.floating]
    keypoints_xyc: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        count = len(self.scores)
        _check_boxes(self.boxes_xyxy, count)
        if self.scores.shape != (count,) or not np.isfinite(self.scores).all():
            raise ValueError("Each box needs a finite detection score")
        if self.keypoints_xyc is not None and self.keypoints_xyc.shape != (
            count,
            NUM_HUMAN_KP,
            3,
        ):
            raise ValueError(
                "Keypoints must be aligned (N,17,3) pixel x/y/confidence rows"
            )


@dataclass(frozen=True)
class ScoringPolicy:
    """Global recall score cutoffs, pose confidence and pseudo-box height strata."""

    score_cutoffs: tuple[float, ...] = (0.25, 0.50, 0.75)
    min_pose_confidence: float = 0.0  # Exclusive. ViTPose confidence is a raw peak.
    pck_cutoffs: tuple[float, ...] = (0.10, 0.20)
    height_cutoffs_px: tuple[float, float] = (80.0, 160.0)

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                (
                    *self.score_cutoffs,
                    self.min_pose_confidence,
                    *self.pck_cutoffs,
                    *self.height_cutoffs_px,
                )
            ).all()
            or len(set(self.score_cutoffs)) != len(self.score_cutoffs)
            or not self.pck_cutoffs
            or len(set(self.pck_cutoffs)) != len(self.pck_cutoffs)
            or min(self.pck_cutoffs) <= 0
            or not 0 < self.height_cutoffs_px[0] < self.height_cutoffs_px[1]
        ):
            raise ValueError("Invalid evaluation cutoffs")


DEFAULT_SCORING_POLICY = ScoringPolicy()
_EMPTY = Predictions(np.empty((0, 4)), np.empty(0))


def describe(values: NDArray[np.floating]) -> dict[str, int | float | None]:
    """Describe finite measurements and retain the count behind the distribution."""
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
    }


def _ratio(count: int, total: int) -> float | None:
    return count / total if total else None


@dataclass(frozen=True)
class _FrameScores:
    reached: NDArray[np.bool_]  # (N, unfiltered + score cutoffs, IoU cutoffs)
    iou: NDArray[np.float64]  # (N,), NaN for unmatched references
    edges: NDArray[np.float64]  # (N,4), signed candidate - reference
    pose_errors: NDArray[np.float64]  # (N,17), NaN for missing predictions
    prediction_count: int


def _score_frame(
    reference: ReferenceFrame, candidate: Predictions, policy: ScoringPolicy
) -> _FrameScores:
    iou_matrix = pairwise_iou(reference.boxes_xyxy, candidate.boxes_xyxy)
    count = len(reference.player_ids)
    reached: NDArray[np.bool_] = np.zeros(
        (count, 1 + len(policy.score_cutoffs), len(IOU_CUTOFFS)), bool
    )
    iou = np.full(count, np.nan)
    edges = np.full((count, 4), np.nan)
    errors = np.full((count, NUM_HUMAN_KP), np.nan)
    for level, score in enumerate((None, *policy.score_cutoffs)):
        selected = (
            np.arange(len(candidate.scores))
            if score is None
            else np.flatnonzero(candidate.scores >= score)
        )
        overlaps = iou_matrix[:, selected]
        for index, cutoff in enumerate(IOU_CUTOFFS):
            refs, columns = match_iou(overlaps, cutoff)
            reached[refs, level, index] = True
            if level != 0 or index != 0:
                continue
            matches = selected[columns]
            iou[refs] = iou_matrix[refs, matches]
            edges[refs] = candidate.boxes_xyxy[matches] - reference.boxes_xyxy[refs]
            if candidate.keypoints_xyc is not None:
                keypoints = candidate.keypoints_xyc[matches].astype(np.float64)
                predicted = np.isfinite(keypoints).all(-1) & (
                    keypoints[..., 2] > policy.min_pose_confidence
                )
                distance = np.linalg.norm(
                    keypoints[..., :2] - reference.keypoints_xy[refs], axis=-1
                )
                errors[refs] = np.where(
                    predicted & reference.joint_valid[refs], distance, np.nan
                )
    return _FrameScores(reached, iou, edges, errors, len(candidate.scores))


def _pose_summary(
    errors: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    areas: NDArray[np.float64],
    policy: ScoringPolicy,
) -> dict[str, Any]:
    valid = eligible & np.isfinite(errors)
    joint_count = int(eligible.sum())
    instance_joints = eligible.sum(-1)
    pose_rows = instance_joints > 0
    # Missing joints contribute zero OKS and fail every PCK cutoff.
    exponent = -(errors**2) / (2 * areas[:, None] * (2 * COCO_SIGMAS) ** 2)
    oks = (
        np.where(valid, np.exp(exponent), 0).sum(-1)[pose_rows]
        / instance_joints[pose_rows]
    )
    relative = errors / np.sqrt(areas[:, None])
    return {
        "pose_references": int(pose_rows.sum()),
        "pose_references_with_prediction": int(valid.any(-1).sum()),
        "eligible_joints": joint_count,
        "predicted_joints": int(valid.sum()),
        "valid_joint_fraction": _ratio(int(valid.sum()), joint_count),
        "oks": describe(oks),
        "pck": {
            str(cutoff): _ratio(int((valid & (relative <= cutoff)).sum()), joint_count)
            for cutoff in policy.pck_cutoffs
        },
        # Pixels have no defined distance for a joint the candidate never returned.
        "pixel_error_px": describe(errors[valid]),
        "by_joint": {
            name: {
                "eligible": int(eligible[:, joint].sum()),
                "pixel_error_px": describe(errors[:, joint][valid[:, joint]]),
            }
            for joint, name in enumerate(COCO_KP_NAMES)
        },
    }


def score_players(
    references: Mapping[FrameKey, ReferenceFrame],
    predictions: Mapping[FrameKey, Predictions],
    *,
    policy: ScoringPolicy = DEFAULT_SCORING_POLICY,
) -> dict[str, Any]:
    """Score every reference image. Omitted candidate frames are detection misses."""
    unexpected = predictions.keys() - references.keys()
    if unexpected:
        raise ValueError(
            f"Prediction key outside the fixed reference images: {min(unexpected)}"
        )
    if not references:
        raise ValueError("Scoring requires at least one reference image")
    keys = sorted(references)
    frames = [references[key] for key in keys]
    scores = [
        _score_frame(frame, predictions.get(key, _EMPTY), policy)
        for key, frame in zip(keys, frames, strict=True)
    ]
    reached = np.concatenate([row.reached for row in scores])
    overlaps = np.concatenate([row.iou for row in scores])
    edges = np.concatenate([row.edges for row in scores])
    errors = np.concatenate([row.pose_errors for row in scores])
    eligible = np.concatenate([frame.joint_valid for frame in frames])
    boxes = np.concatenate([frame.boxes_xyxy for frame in frames]).astype(np.float64)
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(-1)
    heights = boxes[:, 3] - boxes[:, 1]
    n_per_image = [len(frame.player_ids) for frame in frames]
    clips = np.array([key.clip_id for key in keys])
    cameras = np.array([key.camera_id for key in keys])
    row_clip: NDArray[np.str_] = np.repeat(clips, n_per_image)
    row_camera: NDArray[np.str_] = np.repeat(cameras, n_per_image)
    player = np.concatenate([frame.player_ids for frame in frames])

    def summarize(
        mask: NDArray[np.bool_], images: NDArray[np.bool_] | None = None
    ) -> dict[str, Any]:
        count = int(mask.sum())
        matched = mask & np.isfinite(overlaps)
        hits = reached[mask].sum(0)

        def recall(level: int) -> dict[str, float | None]:
            return {
                f"{iou:.2f}": _ratio(int(hits[level, i]), count)
                for i, iou in enumerate(IOU_CUTOFFS)
            }

        primary_hits = int(hits[0, 0])
        return {
            "detection": {
                "references": count,
                "matched_at_iou_0_50": primary_hits,
                "miss_rate_at_iou_0_50": _ratio(count - primary_hits, count),
                "recall_by_iou": recall(0),
                "recall_by_min_score": {
                    str(s): recall(i) for i, s in enumerate(policy.score_cutoffs, 1)
                },
                "matched_iou": describe(overlaps[matched]),
                "xyxy_error_px": {
                    label: {
                        "signed": describe(edges[matched, i]),
                        "absolute": describe(np.abs(edges[matched, i])),
                    }
                    for i, label in enumerate(("x_min", "y_min", "x_max", "y_max"))
                },
                "unmatched_predictions": None
                if images is None
                else (
                    sum(
                        row.prediction_count
                        for row, keep in zip(scores, images, strict=True)
                        if keep
                    )
                    - primary_hits
                ),
                "precision": None,
                "average_precision": None,
                "precision_ap_unavailable_reason": NO_PRECISION_REASON,
            },
            "pose": {
                "matched_only": _pose_summary(
                    errors[matched], eligible[matched], areas[matched], policy
                ),
                "with_detection_coverage": _pose_summary(
                    errors[mask], eligible[mask], areas[mask], policy
                ),
            },
        }

    small, large = policy.height_cutoffs_px
    return {
        "policy": {
            "association": "maximum cardinality above the IoU cutoff, then maximum summed IoU",
            "iou_cutoffs": IOU_CUTOFFS,
            "pose_match_iou": IOU_CUTOFFS[0],
            "pose_confidence_exclusive_cutoff": policy.min_pose_confidence,
            "pck_cutoffs": policy.pck_cutoffs,
            "pck_normalizer": "sqrt(pseudo_bbox_area_px2)",
            "oks_area": "pseudo_bbox_area_px2",
            "oks_sigmas": COCO_SIGMAS.tolist(),
            "height_cutoffs_px": policy.height_cutoffs_px,
            "missing_pose_policy": "joint failures for PCK, zero contributions for OKS, no pixel measurement",
        },
        "images": len(keys),
        "overall": summarize(np.ones(len(player), bool), np.ones(len(keys), bool)),
        "by_clip": {
            str(c): summarize(row_clip == c, clips == c) for c in np.unique(clips)
        },
        "by_camera": {
            str(c): summarize(row_camera == c, cameras == c) for c in np.unique(cameras)
        },
        "by_player": {
            str(c): {
                str(p): summarize((row_clip == c) & (player == p))
                for p in np.unique(player[row_clip == c])
            }
            for c in np.unique(clips)
        },
        "by_size": {
            "small": summarize(heights < small),
            "medium": summarize((heights >= small) & (heights < large)),
            "large": summarize(heights >= large),
        },
    }
