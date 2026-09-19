"""Content identities for immutable reconstruction caches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.reference_pipeline.observations import sha256
from src.tennis_scene.schema import SceneResult

from .checkpoint_integrity import validate_checkpoint_sha256


def validate_teacher_checkpoint_identity(
    identity: Mapping[str, object], expected: Mapping[str, str] | None = None
) -> None:
    """Check recorded teacher identities against pins without rereading weights."""
    checkpoints = identity.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("Teacher identity is missing checkpoints")
    for task in ("plcs", "blcs"):
        digest = checkpoints.get(task)
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"Teacher identity is missing checkpoint: {task}")
        if expected is not None and digest != expected[task]:
            raise ValueError(
                f"Teacher checkpoint pin mismatch: {task}; "
                f"expected {expected[task]}, actual {digest}"
            )


def validate_teacher_checkpoint_metadata(
    scene: SceneResult, identity: Mapping[str, object]
) -> None:
    """Keep raw checkpoint receipts and producer identity consistent, including caches."""
    validate_teacher_checkpoint_identity(identity)
    checkpoints = scene.metadata.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("Raw teacher metadata is missing checkpoints")
    producer = identity["checkpoints"]
    assert isinstance(producer, Mapping)
    for task in ("plcs", "blcs"):
        receipt = checkpoints.get(task)
        digest = receipt.get("sha256") if isinstance(receipt, Mapping) else None
        if digest != producer[task]:
            raise ValueError(
                f"Raw teacher checkpoint mismatch: {task}; "
                f"expected {producer[task]}, actual {digest}"
            )


def scene_identity(
    cfg: DictConfig,
    paths: ReferenceClipPaths,
    clip: ClipManifest,
    observations: Path,
) -> dict[str, Any]:
    """Separate content-changing settings from execution locations and devices."""
    fields = (
        "seed",
        "coordinate_mode",
        "reference_camera",
        "view_half_turns",
        "sample_stride",
        "window_size",
        "window_overlap",
        "pose_visibility_threshold",
        "ball_source",
        "refinement",
    )
    files = [
        "court.npz",
        "court.json",
        "ball_detection_result.json",
        "ball_import.metadata.json",
    ]
    for camera in clip.camera_ids:
        files.extend([f"{camera}_people.npz", f"{camera}_people.metadata.json"])
    checkpoints = {
        task: sha256(getattr(paths, f"{task}_checkpoint")) for task in ("plcs", "blcs")
    }
    pins = (
        validate_checkpoint_sha256(cfg.checkpoint_sha256)
        if "checkpoint_sha256" in cfg
        else None
    )
    validate_teacher_checkpoint_identity({"checkpoints": checkpoints}, pins)
    return {
        "schema_version": 1,
        "clip_manifest_sha256": clip.digest(),
        "settings": OmegaConf.to_container(
            OmegaConf.create({key: cfg[key] for key in fields}), resolve=True
        ),
        "checkpoints": checkpoints,
        "observations": {name: sha256(observations / name) for name in files},
        "video_sha256": {
            camera: sha256(clip.media_path(camera)) for camera in clip.camera_ids
        },
    }


def validated_scene_cache(clip: ClipManifest, identity: dict[str, Any]) -> bool:
    marker = clip.clip_dir / "annotations/tennis_scene/annotation.json"
    if not marker.exists():
        return False
    scene = load_slcs_annotation(clip)
    if scene.metadata.get("dataset_producer_identity") != identity:
        raise ValueError(
            f"Stale reconstruction for {clip.clip_id}; choose a new dataset version"
        )
    validate_teacher_checkpoint_metadata(scene, identity)
    for name in ("human_kp_vis", "court_vis"):
        visibility = getattr(scene, name)
        if (
            visibility is None
            or not np.isfinite(visibility).all()
            or np.any(visibility < 0)
            or np.any(visibility > 1)
        ):
            raise ValueError(
                f"Invalid reconstruction visibility {name} for {clip.clip_id}; "
                "expected finite [0, 1] values; choose a new dataset version and regenerate"
            )
    return True
