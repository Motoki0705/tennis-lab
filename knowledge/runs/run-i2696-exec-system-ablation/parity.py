"""Compare released scene arrays in the same court frame and source timeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from run import CASES, CLIPS, read_json
from scipy.optimize import linear_sum_assignment

from src.tennis_scene.pipeline.dataset_publication import annotation_root
from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path

MATCH_CONFIDENCE = 0.30
MATCH_MIN_FRAMES = 30
MATCH_MAX_DIAGONAL_FRACTION = 0.01


@dataclass(frozen=True, slots=True)
class Scene:
    player_ids: NDArray[np.int32]
    image_wh: tuple[int, int]
    pose_2d: NDArray[np.float32]
    pose_confidence: NDArray[np.float32]
    joints: NDArray[np.float32]
    joint_mask: NDArray[np.bool_]
    smpl_mask: NDArray[np.bool_]


def load_scene(
    dataset: Path, target: dict[str, Any], identity: dict[str, Any]
) -> tuple[dict[str, Any], Scene]:
    record = target["record"]
    clip = record["clip_id"]
    root = annotation_root(dataset / record["path"], identity)
    marker = read_json(root / "annotation.json")
    if (marker["clip_id"], marker["clip_manifest_sha256"], marker["media_sha256"]) != (
        clip,
        target["manifest_sha256"],
        target["media_sha256"],
    ):
        raise ValueError(f"Annotation source hashes differ from the run target: {root}")
    archive = annotation_scene_path(root, marker)
    metadata = read_json(archive.with_suffix(".metadata.json"))
    if (
        metadata["dataset_clip_id"] != clip
        or metadata["camera_ids"] != sorted(target["media_sha256"])
        or metadata["status"] != marker["result_status"]
    ):
        raise ValueError(f"Scene and annotation metadata disagree: {archive}")
    arrays = read_json(root / "scene.json")
    frames = record["num_frames"]
    camera_count = record["num_cameras"]
    with np.load(archive, allow_pickle=False) as data:
        scene = Scene(
            data["player_track_ids"],
            (record["width"], record["height"]),
            data["human_kp_2d"],
            data["human_kp_vis"],
            data["player_kp_3d"],
            data["player_kp_3d_vis"],
            data["player_smpl_valid"],
        )
        if (
            data["ball_3d_valid"].any()
            or marker["validity_statistics"]["ball_3d_frames"] != 0
        ):
            raise ValueError(f"Found a ball in a player-only scene: {archive}")
    expected_shape = (4, camera_count, frames, 17)
    if (
        scene.player_ids.shape != (4,)
        or len(np.unique(scene.player_ids)) != 4
        or scene.player_ids.tolist() != metadata["track_ids"]
        or scene.pose_confidence.shape != expected_shape
        or scene.pose_2d.shape != (*expected_shape, 2)
        or scene.joints.shape != (4, frames, 17, 3)
        or scene.joint_mask.shape != (4, frames, 17)
        or scene.smpl_mask.shape != (4, frames)
        or scene.joint_mask.dtype != bool
        or scene.smpl_mask.dtype != bool
    ):
        raise ValueError(f"Unexpected scene arrays at {archive}")
    if not (
        np.isfinite(scene.pose_confidence).all()
        and np.isfinite(scene.pose_2d[scene.pose_confidence >= MATCH_CONFIDENCE]).all()
        and np.isfinite(scene.joints[scene.joint_mask]).all()
    ):
        raise ValueError(f"Nonfinite valid coordinates at {archive}")
    joint_frames = scene.joint_mask.any(axis=-1).sum(axis=1).tolist()
    smpl_frames = scene.smpl_mask.sum(axis=1).tolist()
    stats = marker["validity_statistics"]
    if (joint_frames, smpl_frames) != (
        stats["player_joint_frames"],
        stats["player_smpl_frames"],
    ):
        raise ValueError(f"Validity counts disagree with the saved masks: {archive}")
    return {
        "scene_status": marker["result_status"],
        "source_frames": frames,
        "player_ids": scene.player_ids.tolist(),
        "player_frames": scene.smpl_mask.size,
        "valid_joint_frames": sum(joint_frames),
        "valid_smpl_frames": sum(smpl_frames),
        "valid_joint_points": int(scene.joint_mask.sum()),
        "valid_joint_frames_by_player": joint_frames,
        "valid_smpl_frames_by_player": smpl_frames,
        "scene_sha256": arrays["exports"]["scene"]["files"]["scene"]["sha256"],
    }, scene


def match_players(
    reference: Scene, candidate: Scene
) -> list[tuple[int, int, float, int, int]]:
    """Match by corresponding camera/time 2D joints, independently of 3D."""
    if (
        reference.image_wh != candidate.image_wh
        or reference.pose_2d.shape[1:] != candidate.pose_2d.shape[1:]
    ):
        raise ValueError(
            "Player matching requires synchronized views and a common image size"
        )
    scale = np.array(reference.image_wh, dtype=np.float64)
    max_distance = MATCH_MAX_DIAGONAL_FRACTION * np.linalg.norm(scale)
    # A rejected match costs more than all accepted distances combined
    # after normalization, so the assignment first matches as many IDs as possible.
    n_reference, n_candidate = len(reference.player_ids), len(candidate.player_ids)
    cost = np.ones((n_reference, n_candidate))
    evidence: dict[tuple[int, int], tuple[float, int, int]] = {}
    for i in range(n_reference):
        for j in range(n_candidate):
            common = (reference.pose_confidence[i] >= MATCH_CONFIDENCE) & (
                candidate.pose_confidence[j] >= MATCH_CONFIDENCE
            )
            frames = int(common.any(axis=(0, 2)).sum())
            if frames < MATCH_MIN_FRAMES:
                continue
            distances = np.linalg.norm(
                (reference.pose_2d[i][common] - candidate.pose_2d[j][common]) * scale,
                axis=-1,
            )
            median_px = float(np.median(distances))
            if median_px <= max_distance:
                cost[i, j] = median_px / max_distance / (n_reference + 1)
                evidence[i, j] = median_px, frames, len(distances)
    rows, cols = linear_sum_assignment(cost)
    return [
        (i, j, *evidence[i, j])
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True)
        if (i, j) in evidence
    ]


def mask_counts(
    reference: NDArray[np.bool_], candidate: NDArray[np.bool_]
) -> dict[str, int]:
    return {
        "both_valid": int((reference & candidate).sum()),
        "reference_only": int((reference & ~candidate).sum()),
        "candidate_only": int((~reference & candidate).sum()),
        "both_invalid": int((~reference & ~candidate).sum()),
    }


def summarize_mask(counts: dict[str, int]) -> dict[str, Any]:
    compared = sum(counts.values())
    return {
        **counts,
        "compared": compared,
        "agreement_fraction": (counts["both_valid"] + counts["both_invalid"]) / compared
        if compared
        else None,
    }


def summarize_distances(distances: NDArray[np.float64]) -> dict[str, Any]:
    return {
        "common_valid_joint_points": len(distances),
        "median_m": float(np.median(distances)) if len(distances) else None,
        "p95_m": float(np.percentile(distances, 95)) if len(distances) else None,
        "max_m": float(distances.max()) if len(distances) else None,
    }


def compare_scenes(
    reference: Scene, candidate: Scene
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    matches = match_players(reference, candidate)
    base_axes = [i for i, _, _, _, _ in matches]
    case_axes = [j for _, j, _, _, _ in matches]
    base_valid, case_valid = (
        reference.joint_mask[base_axes],
        candidate.joint_mask[case_axes],
    )
    common = base_valid & case_valid
    baseline = reference.joints[base_axes][common].astype(np.float64)
    observed = candidate.joints[case_axes][common].astype(np.float64)
    distances = np.linalg.norm(baseline - observed, axis=-1)
    mapping = [
        {
            "baseline_id": int(reference.player_ids[i]),
            "condition_id": int(candidate.player_ids[j]),
            "median_2d_delta_px": median_px,
            "common_2d_frames": frames,
            "common_2d_points": points,
        }
        for i, j, median_px, frames, points in matches
    ]
    return {
        "matched_player_ids": len(mapping),
        "same_numeric_id": sum(
            row["baseline_id"] == row["condition_id"] for row in mapping
        ),
        "player_id_mapping": mapping,
        "joint_point_mask": summarize_mask(mask_counts(base_valid, case_valid)),
        "joint_frame_mask": summarize_mask(
            mask_counts(base_valid.any(axis=-1), case_valid.any(axis=-1))
        ),
        "smpl_mask": summarize_mask(
            mask_counts(reference.smpl_mask[base_axes], candidate.smpl_mask[case_axes])
        ),
        "joint_position_l2": summarize_distances(distances),
    }, distances


def compare_conditions(
    scenes: dict[str, dict[str, tuple[dict[str, Any], Scene]]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for case in CASES:
        per_clip = []
        all_distances = []
        for clip in CLIPS:
            coverage, scene = scenes[case][clip]
            difference, distances = compare_scenes(scenes[CASES[0]][clip][1], scene)
            per_clip.append({"clip_id": clip, **coverage, **difference})
            all_distances.append(distances)
        totals = {
            key: sum(row[key] for row in per_clip)
            for key in (
                "source_frames",
                "player_frames",
                "valid_joint_frames",
                "valid_smpl_frames",
                "valid_joint_points",
                "matched_player_ids",
                "same_numeric_id",
            )
        }
        for name in ("joint_point_mask", "joint_frame_mask", "smpl_mask"):
            totals[name] = summarize_mask(
                {
                    field: sum(row[name][field] for row in per_clip)
                    for field in (
                        "both_valid",
                        "reference_only",
                        "candidate_only",
                        "both_invalid",
                    )
                }
            )
        results[case] = {
            "clips": per_clip,
            "totals": {
                **totals,
                "generated_clips": len(per_clip),
                "joint_position_l2": summarize_distances(np.concatenate(all_distances)),
            },
        }
    return {
        "reference": CASES[0],
        "matching": {
            "min_2d_confidence": MATCH_CONFIDENCE,
            "min_shared_source_frames": MATCH_MIN_FRAMES,
            "max_median_2d_image_diagonal_fraction": MATCH_MAX_DIAGONAL_FRACTION,
        },
        "conditions": results,
    }
