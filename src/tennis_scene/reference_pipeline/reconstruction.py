"""Windowed PLCS/BLCS inference in one explicit reference-court frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io.contracts import BLCSReferenceMetadata
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io.contracts import PLCSReferenceMetadata
from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.tennis_scene.reference_pipeline.observations import (
    court_homographies,
    read_clip,
    sha256,
)
from src.tennis_scene.reference_pipeline.reference import (
    build_reference,
    reference_metadata,
)
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.inference.windowed import blend_windows, window_slices


def associate_people(
    clip: dict[str, Any], output: Path, homographies: np.ndarray, half_turns: list[bool]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep stable track IDs; assign the two singles players by median court end.

    This policy is explicit for singles rallies without an end change. It never
    swaps identity independently per frame. Audit selected IDs and overlay video.
    """
    observations, assignments = [], {}
    for cam, h, turn in zip(clip["camera_ids"], homographies, half_turns, strict=True):
        with np.load(output / f"{cam}_people.npz") as data:
            kp, ids = data["keypoints"], data["track_ids"]
            feet = kp[:, :, [15, 16], :2].mean(axis=2)
            court = cv2.perspectiveTransform(
                feet.reshape(1, -1, 2), np.linalg.inv(h)
            ).reshape(feet.shape)
            if turn:
                court *= -1
            median_y = np.median(court[:, :, 1], axis=1)
            order = np.argsort(median_y)
            if len(order) != 2 or not median_y[order[0]] < 0 < median_y[order[1]]:
                raise ValueError(
                    f"{cam}: selected tracks do not cover both court ends: {median_y}"
                )
            observations.append(kp[order])
            assignments[cam] = {
                "track_ids_near_far_cam0": ids[order].tolist(),
                "median_ground_y_m": median_y[order].tolist(),
                "policy": "fixed singles identity by median cam0-court end",
            }
    return np.stack(observations, axis=1), assignments


def restore_frames(values: np.ndarray, indices: np.ndarray, total: int) -> np.ndarray:
    """Interpolate the time-leading result; hold the final sub-frame explicitly."""
    flat = values.reshape(len(indices), -1)
    result = np.stack(
        [np.interp(np.arange(total), indices, column) for column in flat.T], axis=-1
    )
    return cast(
        np.ndarray, result.reshape((total,) + values.shape[1:]).astype(np.float32)
    )


def reconstruct(cfg: DictConfig, clip_dir: Path, output: Path) -> None:
    clip = read_clip(clip_dir)
    kp, hs = court_homographies(clip_dir)
    aligned, selection, document = build_reference(
        clip["camera_ids"],
        kp,
        list(cfg.view_half_turns),
        cfg.reference_camera,
        (clip["width"], clip["height"]),
    )
    raw, assignments = associate_people(clip, output, hs, list(cfg.view_half_turns))
    (output / "reference_context.json").write_text(json.dumps(document, indent=2))
    (output / "player_association_result.json").write_text(
        json.dumps(assignments, indent=2)
    )
    human_uv = raw[..., :2] / np.asarray([clip["width"], clip["height"]], np.float32)
    confidence = raw[..., 2]
    human_valid = (confidence >= float(cfg.pose_visibility_threshold)) & (
        (human_uv >= 0) & (human_uv <= 1)
    ).all(axis=-1)
    human_uv = np.where(human_valid[..., None], human_uv, 0).astype(np.float32)
    ball = BallDetectionResult.load(output / "ball_detection_result.json")
    total = clip["num_frames"]
    stride = int(cfg.sample_stride)
    if stride <= 0 or int(cfg.window_size) != 128:
        raise ValueError("Require positive stride and the trained 128-frame windows")
    indices = np.arange(0, total, stride)
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": str(Path.cwd()),
            "data_root": str(clip_dir),
            "checkpoint_root": str(Path(cfg.plcs_checkpoint).parent.parent),
            "artifact_root": str(output),
            "output_root": str(output),
            "cache_root": str(output / "cache"),
            "external_asset_root": str(Path(cfg.people.dino_repository).parent),
        },
        repository_root=Path.cwd(),
    )
    resolver = PathResolver(roots)
    plcs = PLCSPredictor.load_from_checkpoint(
        cfg.plcs_checkpoint,
        resolver=resolver,
        device=cfg.device,
        court_keypoint_contract=selection.provenance.contract,
    )
    pmeta = cast(
        PLCSReferenceMetadata, reference_metadata(selection, raw.shape[0], "plcs")
    )
    positions, headings, poses = [], [], []
    windows = window_slices(len(indices), int(cfg.window_size), int(cfg.window_overlap))
    for start, end in windows:
        frame_ids = indices[start:end]
        pred = plcs.predict_multiview_observations(
            human_kp=human_uv[:, :, frame_ids],
            human_vis=human_valid[:, :, frame_ids],
            court_kp=aligned[:, frame_ids],
            court_vis=np.ones(aligned[:, frame_ids].shape[:-1], bool),
            padding_mask=np.zeros((raw.shape[0], len(kp), len(frame_ids)), bool),
            court_keypoint_metadata=document,
            reference_metadata=pmeta,
        )
        if pred.canonical_pose is None:
            raise ValueError("Canonical-pose prediction required")
        positions.append((start, pred.position_meters.transpose(1, 0, 2)))
        headings.append(
            (
                start,
                np.stack(
                    [np.cos(pred.yaw_radians), np.sin(pred.yaw_radians)], -1
                ).transpose(1, 0, 2),
            )
        )
        poses.append((start, pred.canonical_pose.transpose(1, 0, 2, 3)))
        print(f"PLCS window {start}:{end}", flush=True)
    pos = restore_frames(
        blend_windows(positions, len(indices)), indices, total
    ).transpose(1, 0, 2)
    heading = restore_frames(
        blend_windows(headings, len(indices)), indices, total
    ).transpose(1, 0, 2)
    yaw = np.arctan2(heading[..., 1], heading[..., 0]).astype(np.float32)
    canonical = restore_frames(
        blend_windows(poses, len(indices)), indices, total
    ).transpose(1, 0, 2, 3)
    joints = canonical.copy()
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    joints[..., 0] = canonical[..., 0] * c - canonical[..., 1] * s
    joints[..., 1] = canonical[..., 0] * s + canonical[..., 1] * c
    joints += pos[:, :, None]
    del plcs
    torch.cuda.empty_cache()
    blcs = BLCSPredictor.load_from_checkpoint(
        cfg.blcs_checkpoint,
        resolver=resolver,
        device=cfg.device,
        court_keypoints=selection.provenance.contract,
    )
    bmeta = cast(BLCSReferenceMetadata, reference_metadata(selection, 1, "blcs"))
    trajectories = []
    for start, end in windows:
        f = indices[start:end]
        pred_ball = blcs.predict_multiview_arrays(
            ball_uv=ball.ball_uv[:, f],
            ball_vis=ball.visibility[:, f],
            court_kp=aligned[:, f],
            court_vis=np.ones(aligned[:, f].shape[:-1], bool),
            denormalize=True,
            court_keypoint_document=document,
            court_reference_provenance=(selection.provenance,),
            reference_metadata=bmeta,
        )
        # cam0 defines identity transform, asserted by build_reference.
        trajectories.append((start, pred_ball.position.squeeze(0).cpu().numpy()))
        print(f"BLCS window {start}:{end}", flush=True)
    trajectory = restore_frames(
        blend_windows(trajectories, len(indices)), indices, total
    )
    for name, value in [
        ("player position", pos),
        ("joints", joints),
        ("ball", trajectory),
    ]:
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite {name} predictions")
    metadata = {
        "schema_version": 1,
        "pipeline": "dino_vitpose_axial_reference",
        "reference": document,
        "person_association": assignments,
        "pose_source": "PLCS TemporalDecomposedCanonicalPoseHead",
        "smpl_available": False,
        "court_observations": {
            "manual_slots": 14,
            "plcs_slots": 14,
            "blcs_slots": 14,
            "blcs_unobserved_slots": [],
        },
        "sample_stride": stride,
        "inference_fps": clip["fps"] / stride,
        "time_restore": "linear positions/canonical joints; circular heading; final sub-frame held",
        "checkpoints": {
            task: {
                "path": str(cfg[f"{task}_checkpoint"]),
                "sha256": sha256(Path(cfg[f"{task}_checkpoint"])),
            }
            for task in ("plcs", "blcs")
        },
        "ball_input_statuses": ["observed", "interpolated", "occlusion_estimated"],
    }
    result = SceneResult(
        total,
        clip["fps"],
        clip["width"],
        clip["height"],
        aligned,
        np.ones(aligned.shape[:-1], np.float32),
        pos,
        yaw,
        ball_uv=ball.ball_uv,
        ball_vis=ball.visibility,
        ball_3d=trajectory,
        human_kp_2d=human_uv.astype(np.float32),
        human_kp_vis=np.where(human_valid, confidence, 0).astype(np.float32),
        player_track_ids=np.arange(raw.shape[0], dtype=np.int32),
        player_kp_3d=joints,
        player_canonical_pose=canonical,
        metadata=metadata,
    )
    save_scene_result(result, output / "scene.npz")
    print("Saved scene.npz", flush=True)
