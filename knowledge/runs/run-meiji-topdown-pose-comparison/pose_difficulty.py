"""Measure pose accuracy and temporal residuals in fixed difficulty groups."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from evaluate import POLICY
from fixed_inputs import RowKey, row_keys
from numpy.typing import NDArray
from pose_contract import PoseBatch
from replay_accuracy import read_condition, row_columns

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.reference import BoxQuality, JointQuality
from src.tennis_scene.evaluation.player_pose.scoring import (
    COCO_SIGMAS,
    FrameKey,
    ReferenceFrame,
    describe,
)
from src.utils.geometry.bbox import pairwise_iou
from src.utils.io import write_json_atomic
from src.utils.schema.player import COCO_KP_IDX, COCO_KP_NAMES

SMALL_HEIGHT = POLICY.height_cutoffs_px[0]
EDGE_DISTANCE = 10
OVERLAP_IOU = 0.10
WRIST_SPEED_PERCENTILE = 90
JITTER_FRAME_STEP = 2
OKS_CUTOFFS = (0.50, 0.75)


@dataclass(frozen=True)
class Difficulty:
    xy: NDArray[np.float64]
    eligible: NDArray[np.bool_]
    box_area: NDArray[np.float64]
    groups: Mapping[str, NDArray[np.bool_]]
    triplets: NDArray[np.intp]
    evidence: dict[str, Any]


def wrist_speeds(
    joints: NDArray[np.floating],
    inferred: NDArray[np.bool_],
    frames: NDArray[np.integer],
    fps: float,
) -> NDArray[np.float64]:
    """Maximum wrist speed available for each player and full source frame."""
    wrist_indices = [COCO_KP_IDX[name] for name in ("left_wrist", "right_wrist")]
    wrist = joints[:, :, wrist_indices, :].astype(np.float64)
    before, after = frames[:-1], frames[1:]
    step = after - before
    valid_pair = inferred[:, before] & inferred[:, after] & (step <= 2) & (step > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = np.linalg.norm(wrist[:, after] - wrist[:, before], axis=-1)
        velocity *= fps / step[None, :, None]
    velocity[~np.isfinite(velocity)] = np.nan
    maximum = np.fmax(velocity[..., 0], velocity[..., 1])
    speed = np.full(inferred.shape, np.nan, np.float64)
    speed[:, after] = np.where(valid_pair & np.isfinite(maximum), maximum, np.nan)
    return speed


def temporal_triplets(keys: tuple[RowKey, ...]) -> NDArray[np.intp]:
    """Find centered windows at t-2, t, t+2 within a clip, camera and player."""
    index = {row: i for i, row in enumerate(keys)}
    triplets = []
    for center, row in enumerate(keys):
        clip, camera, frame = row.frame
        before = RowKey(
            FrameKey(clip, camera, frame - JITTER_FRAME_STEP), row.player_id
        )
        after = RowKey(FrameKey(clip, camera, frame + JITTER_FRAME_STEP), row.player_id)
        if before in index and after in index:
            triplets.append((index[before], center, index[after]))
    return np.asarray(triplets, dtype=np.intp).reshape(-1, 3)


def prepare_difficulty(
    bundle: Path, manifest: dict[str, Any], references: dict[FrameKey, ReferenceFrame]
) -> Difficulty:
    keys = row_keys(references)
    ordered = [references[key] for key in sorted(references)]
    boxes = np.concatenate([frame.boxes_xyxy for frame in ordered]).astype(np.float64)
    xy = np.concatenate([frame.keypoints_xy for frame in ordered]).astype(np.float64)
    eligible = np.concatenate([frame.joint_valid for frame in ordered])
    widths, heights = (boxes[:, 2:] - boxes[:, :2]).T
    clip_ids = np.array([row.frame.clip_id for row in keys])
    edge: NDArray[np.bool_] = np.zeros(len(keys), bool)
    occluded: NDArray[np.bool_] = np.zeros(len(keys), bool)
    speed: NDArray[np.float64] = np.full(len(keys), np.nan, np.float64)
    clip_speeds = []
    quality = {}
    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]
        rows = np.flatnonzero(clip_ids == clip_id)
        source = pinned_file(bundle, clip["pseudo_ground_truth"])
        with np.load(source, allow_pickle=False) as data:
            players = {int(player): i for i, player in enumerate(data["player_ids"])}
            cameras = {str(cam): i for i, cam in enumerate(data["camera_ids"])}
            player = [players[keys[i].player_id] for i in rows]
            camera = [cameras[keys[i].frame.camera_id] for i in rows]
            frames = [keys[i].frame.source_frame for i in rows]
            idx = player, camera, frames
            box_reasons = data["box_reasons"]
            joint_reasons = data["joint_reasons"]
            if not np.all(
                box_reasons[idx] == BoxQuality.ELIGIBLE
            ) or not np.array_equal(
                joint_reasons[idx] == JointQuality.ELIGIBLE, eligible[rows]
            ):
                raise ValueError(
                    f"Difficulty rows differ from the scoring reference for {clip_id}"
                )
            occluded[rows] = data["likely_inter_player_occlusion"][idx].any(axis=-1)
            eval_frames = data["evaluation_frames"]
            court_speed = wrist_speeds(
                data["joints_court_m"],
                data["gvhmr_inferred"],
                eval_frames,
                clip["fps"],
            )
            clip_speeds.append(court_speed[np.isfinite(court_speed)])
            speed[rows] = court_speed[player, frames]
            box_codes = box_reasons[:, :, eval_frames]
            joint_codes = joint_reasons[:, :, eval_frames]
            quality[clip_id] = {
                "box_reasons": {
                    reason.name.lower(): int((box_codes == reason).sum())
                    for reason in BoxQuality
                },
                "joint_reasons": {
                    reason.name.lower(): (joint_codes == reason)
                    .sum(axis=(0, 1, 2))
                    .tolist()
                    for reason in JointQuality
                },
            }
        selected = boxes[rows]
        image_size = np.array([clip["width"], clip["height"]], dtype=np.float64)
        edge[rows] = np.any(
            np.column_stack((selected[:, :2], image_size - selected[:, 2:]))
            <= EDGE_DISTANCE,
            axis=1,
        )

    def overlap(frame: ReferenceFrame) -> NDArray[np.bool_]:
        iou = pairwise_iou(frame.boxes_xyxy, frame.boxes_xyxy)
        np.fill_diagonal(iou, 0)
        return (iou > OVERLAP_IOU).any(axis=1)

    speed_samples = np.concatenate(clip_speeds)
    if not speed_samples.size:
        raise ValueError("The reference has no valid wrist speed samples")
    cutoff = float(np.percentile(speed_samples, WRIST_SPEED_PERCENTILE))
    groups = {
        "all": np.ones(len(keys), bool),
        "small_projected_person": heights < SMALL_HEIGHT,
        "fast_wrist": speed >= cutoff,
        "overlapping_boxes": np.concatenate([overlap(frame) for frame in ordered]),
        "likely_inter_player_occlusion": occluded,
        "image_edge": edge,
        "incomplete_reference": ~eligible.all(axis=1),
    }
    triplets = temporal_triplets(keys)
    return Difficulty(
        xy,
        eligible,
        widths * heights,
        groups,
        triplets,
        {
            "small_height_px_exclusive": SMALL_HEIGHT,
            "edge_distance_px_inclusive": EDGE_DISTANCE,
            "overlap_iou_exclusive": OVERLAP_IOU,
            "fast_percentile": WRIST_SPEED_PERCENTILE,
            "fast_wrist_cutoff_mps": cutoff,
            "unique_wrist_speed_samples": int(speed_samples.size),
            "rows_with_unknown_wrist_speed": int((~np.isfinite(speed)).sum()),
            "jitter_frame_step": JITTER_FRAME_STEP,
            "joint_order": COCO_KP_NAMES,
            "quality_by_clip": quality,
        },
    )


def _fraction(count: int, total: int) -> float | None:
    return count / total if total else None


def joint_metrics(cohort: Difficulty, poses: PoseBatch) -> dict[str, Any]:
    """Score fixed target joints and count missing predictions for each group."""
    eligible = cohort.eligible
    model = poses.keypoints_xyc.astype(np.float64)
    valid = (
        eligible
        & np.isfinite(model).all(axis=-1)
        & (model[..., 2] > POLICY.min_pose_confidence)
    )
    residual = np.full_like(cohort.xy, np.nan)
    residual[valid] = model[..., :2][valid] - cohort.xy[valid]
    error = np.linalg.norm(residual, axis=-1)
    relative = error / np.sqrt(cohort.box_area[:, None])
    passes = {str(threshold): relative <= threshold for threshold in POLICY.pck_cutoffs}
    exponent = -(error**2) / (2 * cohort.box_area[:, None] * (2 * COCO_SIGMAS) ** 2)
    joint_counts = eligible.sum(axis=1)
    row_oks: NDArray[np.float64] = np.full(len(eligible), np.nan, np.float64)
    np.divide(
        np.where(valid, np.exp(exponent), 0.0).sum(axis=1),
        joint_counts,
        out=row_oks,
        where=joint_counts > 0,
    )
    before, center, after = cohort.triplets.T
    jitter = np.full_like(error, np.nan)
    jitter[center] = np.linalg.norm(
        residual[before] - 2 * residual[center] + residual[after], axis=-1
    )
    jitter_eligible = np.zeros_like(eligible)
    jitter_eligible[center] = eligible[before] & eligible[center] & eligible[after]

    def summarize(row_mask: NDArray[np.bool_], index: slice | int) -> dict[str, Any]:
        select = (row_mask, index)
        count = int(eligible[select].sum())
        predicted = int(valid[select].sum())
        return {
            "eligible_joints": count,
            "predicted_joints": predicted,
            "missing_joints": count - predicted,
            "coverage": _fraction(predicted, count),
            "pixel_error_px": describe(error[select]),
            "pck": {
                threshold: _fraction(int(hit[select].sum()), count)
                for threshold, hit in passes.items()
            },
            "eligible_temporal_triplets": int(jitter_eligible[select].sum()),
            "jitter_px": describe(jitter[select]),
        }

    results = {}
    for name, mask in cohort.groups.items():
        oks = row_oks[mask & np.isfinite(row_oks)]
        results[name] = {
            "person_rows": int(mask.sum()),
            **summarize(mask, slice(None)),
            "oks": describe(oks),
            "oks_success": {
                str(threshold): _fraction(int((oks >= threshold).sum()), len(oks))
                for threshold in OKS_CUTOFFS
            },
            "by_joint": {
                joint: summarize(mask, index)
                for index, joint in enumerate(COCO_KP_NAMES)
            },
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = load_manifest(args.bundle, expected_sha256=args.bundle_sha256)
    bundle = manifest["benchmark_id"], args.bundle_sha256
    references = load_reference_frames(args.bundle, expected_sha256=args.bundle_sha256)
    columns = row_columns(references, row_keys(references))
    cohort = prepare_difficulty(args.bundle, manifest, references)
    conditions = {}
    for job in args.jobs:
        (model, backend), poses, previous, record = read_condition(
            args.queue, job, bundle, columns
        )
        key = f"{model}/{backend}"
        if key in conditions:
            raise ValueError(f"Duplicate diagnosis for {key}")
        metrics = joint_metrics(cohort, poses)
        canonical = previous["overall"]["pose"]["with_detection_coverage"]
        overall = metrics["all"]
        if (
            canonical["pck"] != overall["pck"]
            or canonical["eligible_joints"] != overall["eligible_joints"]
            or canonical["predicted_joints"] != overall["predicted_joints"]
            or not np.isclose(
                canonical["oks"]["mean"], overall["oks"]["mean"], atol=1e-12, rtol=0
            )
        ):
            raise ValueError(
                f"Diagnostics differ from canonical overall metrics for {key}"
            )
        conditions[key] = {
            "job_id": job,
            "predictions_sha256": record["prediction_sha256"],
            "groups": metrics,
        }
    write_json_atomic(
        args.output,
        {
            "bundle_id": bundle[0],
            "bundle_sha256": bundle[1],
            "policy": cohort.evidence,
            "conditions": conditions,
        },
    )


if __name__ == "__main__":
    main()
