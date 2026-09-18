"""Content identities for immutable reconstruction caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tennis_scene.configuration import ReferenceClipPaths
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.reference_pipeline.observations import sha256


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
    files = ["court.npz", "court.json", "ball_detection_result.json", "ball_import.metadata.json"]
    for camera in clip.camera_ids:
        files.extend([f"{camera}_people.npz", f"{camera}_people.metadata.json"])
    return {
        "schema_version": 1,
        "clip_manifest_sha256": clip.digest(),
        "settings": OmegaConf.to_container(
            OmegaConf.create({key: cfg[key] for key in fields}), resolve=True
        ),
        "checkpoints": {
            task: sha256(getattr(paths, f"{task}_checkpoint"))
            for task in ("plcs", "blcs")
        },
        "observations": {name: sha256(observations / name) for name in files},
        "video_sha256": {camera: sha256(clip.media_path(camera)) for camera in clip.camera_ids},
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
