"""Explicit, non-destructive import of legacy broadcast media and 2D balls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetClipRecord,
    DatasetManifest,
    validate_id_component,
)
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.io import save_json_atomic


def filter_saved_ball(
    uv: np.ndarray,
    visibility: np.ndarray,
    *,
    isolated_jump: float,
    neighbor_distance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mask isolated spikes, retaining missing frames instead of inventing labels.

    Both neighbours must be observed and mutually close. A genuine large
    displacement shared by subsequent frames is not rejected by this test.
    Distances are normalized image UV, not uncalibrated physical velocities.
    """
    if uv.ndim != 3 or uv.shape[-1] != 2 or visibility.shape != uv.shape[:-1]:
        raise ValueError("Saved ball arrays must be (V,T,2) and (V,T)")
    if visibility.dtype != np.bool_:
        raise TypeError("Saved ball visibility must be boolean")
    if not 0 < neighbor_distance < isolated_jump < 1:
        raise ValueError("Require 0 < neighbor_distance < isolated_jump < 1")
    finite_inside = np.isfinite(uv).all(-1) & ((uv >= 0) & (uv <= 1)).all(-1)
    if (visibility & ~finite_inside).any():
        raise ValueError("Visible saved ball coordinates are not finite normalized UV")
    valid = visibility.copy()
    all_three = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    isolated = (
        all_three
        & (np.linalg.norm(uv[:, 1:-1] - uv[:, :-2], axis=-1) > isolated_jump)
        & (np.linalg.norm(uv[:, 1:-1] - uv[:, 2:], axis=-1) > isolated_jump)
        & (np.linalg.norm(uv[:, 2:] - uv[:, :-2], axis=-1) < neighbor_distance)
    )
    valid[:, 1:-1] &= ~isolated
    return valid, {
        "policy": "reject isolated UV spikes with two close observed neighbours; no interpolation",
        "isolated_jump_uv": isolated_jump,
        "neighbor_distance_uv": neighbor_distance,
        "source_observed": int(visibility.sum()),
        "rejected_spikes": int(isolated.sum()),
        "retained": int(valid.sum()),
        "coverage": valid.mean(axis=1).tolist(),
        "is_ground_truth": False,
    }


def copy_legacy_broadcast(
    source: Path,
    destination: Path,
    *,
    dataset_id: str,
    video_groups: dict[str, str],
    isolated_jump: float,
    neighbor_distance: float,
) -> None:
    """Copy v1 -> v2 into a new version and record every source hash.

    ``video_groups`` is a curated match/venue grouping of disjoint clips from a
    compilation, never inferred from model predictions. Media is hard-linked;
    old PLCS, BLCS, pose and court predictions are not imported as new labels.
    """
    if (
        source.resolve() == destination.resolve()
        or destination.resolve().is_relative_to(source.resolve())
        or source.resolve().is_relative_to(destination.resolve())
    ):
        raise ValueError("Legacy import requires an independent destination")
    validate_id_component(dataset_id, field_name="dataset_id")
    root = json.loads((source / "dataset.json").read_text())
    if root.get("version") != 1 or set(root) != {
        "version",
        "created_at",
        "updated_at",
        "clips",
    }:
        raise ValueError(
            "Explicit legacy import requires the recording-based v1 dataset"
        )
    if set(video_groups) != {record["clip_id"] for record in root["clips"]}:
        raise ValueError("Curated video_groups must cover every legacy clip exactly")
    input_hashes = {}
    for record in root["clips"]:
        if (
            len(Path(record["clip_id"]).parts) != 2
            or any(part in {"..", "."} for part in Path(record["clip_id"]).parts)
            or Path(record["clip_id"]).is_absolute()
        ):
            raise ValueError("Invalid legacy clip ID")
        if record["path"] != f"clips/{record['clip_id']}":
            raise ValueError("Invalid legacy clip path")
        old = source / record["path"]
        raw = json.loads((old / "clip.json").read_text())
        inputs = [
            "clip.json",
            "annotations/tennis_scene/annotation.json",
            "annotations/tennis_scene/scene.npz",
            *raw["video_paths"],
        ]
        for name in inputs:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Legacy media path escapes its clip")
        input_hashes[record["clip_id"]] = {name: sha256(old / name) for name in inputs}
    settings = {
        "source_manifest_sha256": sha256(source / "dataset.json"),
        "dataset_id": dataset_id,
        "video_groups": video_groups,
        "isolated_jump_uv": isolated_jump,
        "neighbor_distance_uv": neighbor_distance,
        "inputs": input_hashes,
    }
    receipt = destination / "legacy_import.json"
    if receipt.exists() and json.loads(receipt.read_text())["settings"] != settings:
        raise ValueError("Legacy import recipe changed; choose a new version")
    index = DatasetManifest(
        dataset_id, created_at=root["created_at"], updated_at=root["updated_at"]
    )
    provenance = {}
    for record in root["clips"]:
        old_id = record["clip_id"]
        if record["path"] != f"clips/{old_id}":
            raise ValueError(f"Invalid legacy clip path: {record['path']}")
        old = source / record["path"]
        raw = json.loads((old / "clip.json").read_text())
        if raw.get("version") != 1 or raw.get("clip_id") != old_id:
            raise ValueError(f"Legacy inventory/manifest mismatch: {old_id}")
        marker = json.loads(
            (old / "annotations/tennis_scene/annotation.json").read_text()
        )
        if marker["clip_manifest_sha256"] != sha256(old / "clip.json"):
            raise ValueError(f"Stale legacy scene observations: {old_id}")
        group = validate_id_component(video_groups[old_id], field_name="video_group")
        clip_id = f"{group}/{raw['clip_name']}"
        relative = f"videos/{group}/clips/{raw['clip_name']}"
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        migrated = {
            **raw,
            "version": 2,
            "dataset_id": dataset_id,
            "video_id": group,
            "clip_id": clip_id,
        }
        del migrated["recording_id"]
        target_manifest = target / "clip.json"
        if (
            target_manifest.exists()
            and json.loads(target_manifest.read_text()) != migrated
        ):
            raise ValueError(f"Changed legacy clip {old_id}")
        save_json_atomic(migrated, target_manifest)
        clip = ClipManifest.load(target)
        videos = {}
        for video_name in clip.video_paths:
            source_video, target_video = old / video_name, target / video_name
            videos[video_name] = sha256(source_video)
            target_video.parent.mkdir(parents=True, exist_ok=True)
            if target_video.exists():
                if sha256(target_video) != videos[video_name]:
                    raise ValueError(f"Changed imported media {target_video}")
            else:
                os.link(source_video, target_video)
        scene_path = old / "annotations/tennis_scene/scene.npz"
        with np.load(scene_path, allow_pickle=False) as scene:
            uv, visibility = scene["ball_uv"], scene["ball_vis"]
            if uv.shape != (len(clip.camera_ids), clip.num_frames, 2):
                raise ValueError(f"Legacy ball shape differs from manifest: {old_id}")
        valid, report = filter_saved_ball(
            uv,
            visibility,
            isolated_jump=isolated_jump,
            neighbor_distance=neighbor_distance,
        )
        observation = target / "observations"
        observation.mkdir(exist_ok=True)
        ball = BallDetectionResult(
            np.where(valid[..., None], uv, 0).astype(np.float32),
            np.where(valid[..., None], uv * [clip.width, clip.height], 0).astype(
                np.float32
            ),
            valid,
            valid.astype(np.float32),
        )
        ball.save(observation / "ball_detection_result.json")
        identity = {
            "legacy_clip_id": old_id,
            "legacy_manifest_sha256": sha256(old / "clip.json"),
            "scene_sha256": sha256(scene_path),
            "video_sha256": videos,
            "source_path": str(scene_path.resolve()),
            "quality": report,
        }
        save_json_atomic(identity, observation / "ball_import.metadata.json")
        provenance[clip_id] = identity
        index.clips[clip_id] = DatasetClipRecord(
            clip_id,
            group,
            clip.clip_name,
            relative,
            len(clip.camera_ids),
            clip.num_frames,
            clip.fps,
            clip.width,
            clip.height,
        )
    index.save(destination)
    save_json_atomic({"settings": settings, "clips": provenance}, receipt)
