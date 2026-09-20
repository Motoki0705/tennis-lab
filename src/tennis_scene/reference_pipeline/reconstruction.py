"""Windowed PLCS/BLCS inference in one explicit reference-court frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.base.generate_dataset import (
    build_physical_court_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io.contracts import BLCSReferenceMetadata
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io.contracts import PLCSReferenceMetadata
from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.tennis_scene.reference_pipeline.observations import (
    court_homographies,
    read_clip,
    sha256,
)
from src.tennis_scene.reference_pipeline.reference import (
    build_reference,
    fit_camera,
    reference_metadata,
)
from src.tennis_scene.schema import SceneResult
from src.utils.inference.windowed import (
    blend_windows,
    restore_sampled_frames,
    window_slices,
)


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
            if not np.isfinite(kp[..., 2]).all():
                raise ValueError(f"{cam}: non-finite ViTPose heatmap peaks")
            if "pose_supported_mask" in data:
                supported = data["pose_supported_mask"]
                if supported.shape != kp.shape[:2] or supported.dtype != np.bool_:
                    raise ValueError(f"{cam}: invalid pose_supported_mask")
                kp[..., 2] = np.where(supported[..., None], kp[..., 2], 0)
            valid_feet = (kp[:, :, [15, 16], 2] > 0).all(axis=2)
            valid_feet &= np.isfinite(kp[:, :, [15, 16], :]).all(axis=(2, 3))
            if not valid_feet.any(axis=1).all():
                raise ValueError(f"{cam}: selected tracks have no supported feet")
            feet = kp[:, :, [15, 16], :2].mean(axis=2)
            court = cv2.perspectiveTransform(
                feet.reshape(1, -1, 2), np.linalg.inv(h)
            ).reshape(feet.shape)
            if turn:
                court *= -1
            median_y = np.array(
                [
                    np.median(points[valid, 1])
                    for points, valid in zip(court, valid_feet, strict=True)
                ]
            )
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


def pose_visibility_from_heatmap_peaks(
    peaks: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Explicitly bound unnormalized ViTPose peaks to the visibility contract."""
    if not peaks.size or not np.isfinite(peaks).all():
        raise ValueError("ViTPose heatmap peaks must be nonempty and finite")
    audit = {
        "method": "clip_heatmap_peak_to_unit_interval",
        "input_semantics": "unnormalized ViTPose heatmap peak; not a probability",
        "raw_min": float(peaks.min()),
        "raw_max": float(peaks.max()),
        "saturated_below_zero_count": int(np.count_nonzero(peaks < 0)),
        "saturated_above_one_count": int(np.count_nonzero(peaks > 1)),
    }
    return np.clip(peaks, 0, 1).astype(np.float32), audit


def human_observation_mask(
    human_uv: np.ndarray, confidence: np.ndarray, threshold: float
) -> np.ndarray:
    """Zero confidence always means missing, including at a zero threshold."""
    valid: np.ndarray = (
        (confidence > 0)
        & (confidence >= threshold)
        & ((human_uv >= 0) & (human_uv <= 1)).all(axis=-1)
    )
    return valid


def restore_frames(values: np.ndarray, indices: np.ndarray, total: int) -> np.ndarray:
    """Interpolate the time-leading result; hold the final sub-frame explicitly."""
    restored: np.ndarray = restore_sampled_frames(values, indices, total).astype(
        np.float32
    )
    return restored


def reconstruct(
    cfg: DictConfig,
    paths: ReferenceClipPaths,
    clip_dir: Path,
    output: Path,
    *,
    court_observations: tuple[np.ndarray, np.ndarray] | None = None,
    observation_directory: Path | None = None,
    coordinate_mode: Literal["reference", "physical"] = "reference",
) -> SceneResult:
    if coordinate_mode not in {"reference", "physical"}:
        raise ValueError("Explicit reference or physical coordinate_mode required")
    clip = read_clip(clip_dir, minimum_views=3 if coordinate_mode == "reference" else 1)
    kp, hs = (
        court_homographies(clip_dir)
        if court_observations is None
        else court_observations
    )
    selection = None
    if coordinate_mode == "reference":
        half_turns = list(cfg.view_half_turns)
        aligned, selection, document = build_reference(
            clip["camera_ids"],
            kp,
            half_turns,
            cfg.reference_camera,
            (clip["width"], clip["height"]),
        )
        provenance = selection.provenance
        model_document = document
    else:
        if cfg.reference_camera is not None or cfg.view_half_turns is not None:
            raise ValueError("physical_v1 forbids reference-camera declarations")
        if len(kp) != 1:
            raise ValueError(
                "Physical broadcast mode currently requires one fixed camera"
            )
        half_turns = [False]
        aligned = kp
        provenance = build_physical_court_provenance()
        model_document = None
        document = {
            "camera_ids": clip["camera_ids"],
            "coordinate_mode": "physical_v1",
            "camera_fits": [
                fit_camera(kp[0, 0], (clip["width"], clip["height"]), False)
            ],
            "court_reference_provenance": provenance.to_dict(),
        }
    contract = resolve_court_keypoint_contract(
        "camera_view_v2" if coordinate_mode == "reference" else "physical_v1"
    )
    observed = output if observation_directory is None else observation_directory
    raw, assignments = associate_people(clip, observed, hs, half_turns)
    (output / "reference_context.json").write_text(json.dumps(document, indent=2))
    (output / "player_association_result.json").write_text(
        json.dumps(assignments, indent=2)
    )
    human_uv = raw[..., :2] / np.asarray([clip["width"], clip["height"]], np.float32)
    confidence, pose_visibility_audit = pose_visibility_from_heatmap_peaks(raw[..., 2])
    human_valid = human_observation_mask(
        human_uv, confidence, float(cfg.pose_visibility_threshold)
    )
    human_uv = np.where(human_valid[..., None], human_uv, 0).astype(np.float32)
    ball = BallDetectionResult.load(observed / "ball_detection_result.json")
    total = clip["num_frames"]
    stride = int(cfg.sample_stride)
    if stride <= 0 or int(cfg.window_size) != 128:
        raise ValueError("Require positive stride and the trained 128-frame windows")
    indices = np.arange(0, total, stride)
    plcs = PLCSPredictor.load_from_checkpoint(
        paths.plcs_checkpoint,
        resolver=paths.resolver,
        device=cfg.device,
        court_keypoint_contract=contract,
    )
    pmeta = (
        cast(PLCSReferenceMetadata, reference_metadata(selection, raw.shape[0], "plcs"))
        if selection is not None
        else None
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
            court_keypoint_metadata=model_document,
            court_reference_provenance=(provenance,) * raw.shape[0],
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
        paths.blcs_checkpoint,
        resolver=paths.resolver,
        device=cfg.device,
        court_keypoints=contract,
    )
    bmeta = (
        cast(BLCSReferenceMetadata, reference_metadata(selection, 1, "blcs"))
        if selection is not None
        else None
    )
    trajectories = []
    for start, end in windows:
        f = indices[start:end]
        pred_ball = blcs.predict_multiview_arrays(
            ball_uv=ball.ball_uv[:, f],
            ball_vis=ball.visibility[:, f],
            court_kp=aligned[:, f],
            court_vis=np.ones(aligned[:, f].shape[:-1], bool),
            denormalize=True,
            court_keypoint_document=model_document,
            court_reference_provenance=(provenance,),
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
        "pipeline": f"dino_vitpose_{coordinate_mode}",
        "reference": document,
        "person_association": assignments,
        "pose_visibility_conversion": pose_visibility_audit,
        "pose_source": "configured PLCS canonical pose head",
        "smpl_available": False,
        "court_observations": {
            "source": "manual"
            if court_observations is None
            else "configured_static_court_model",
            "observed_slots": 14,
            "plcs_slots": 14,
            "blcs_slots": 14,
            "blcs_unobserved_slots": [],
        },
        "sample_stride": stride,
        "inference_fps": clip["fps"] / stride,
        "time_restore": "linear positions/canonical joints; circular heading; final sub-frame held",
        "checkpoints": {
            task: {
                "path": str(getattr(paths, f"{task}_checkpoint")),
                "sha256": sha256(getattr(paths, f"{task}_checkpoint")),
            }
            for task in ("plcs", "blcs")
        },
        "ball_input_provenance": json.loads(
            (observed / "ball_import.metadata.json").read_text()
        ),
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
    return result
