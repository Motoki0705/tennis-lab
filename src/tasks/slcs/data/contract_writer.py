"""Test-data writer for the canonical tennis-scene dataset contract.

This is the executable counterpart of :mod:`src.tasks.slcs.data.contract`.
It exists so that

- tests and smoke runs can materialize a contract-conformant dataset, and
- smoke tests can exercise the same manifests produced by Clip Studio and
  ``generate_dataset``.

It follows the same failure rules as the reader: id collisions, mismatched
overwrites and partially written annotations are explicit errors, and the
``annotation.json`` completion marker is always written last.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.slcs.data.contract import (
    ANNOTATION_MARKER_NAME,
    CLIP_MANIFEST_NAME,
    CLIPS_DIR_NAME,
    MEDIA_DIR_NAME,
    SCENE_NPZ_NAME,
    ClipManifest,
    DatasetContractError,
    tennis_scene_dir,
    validate_id_component,
)
from src.tennis_scene.generate_dataset.manifest import (
    CLIP_SCHEMA_VERSION,
    register_exported_clip,
)
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    ANNOTATION_SCHEMA_VERSION,
)
from src.tennis_scene.io import SceneResult
from src.utils.io import ensure_dir, save_json, save_json_atomic, utc_now_iso
from src.utils.video.writer import save_video_rgb


def write_clip_manifest(
    dataset_root: str | Path,
    *,
    recording_id: str,
    clip_name: str,
    fps: float,
    num_frames: int,
    width: int,
    height: int,
    media_videos: dict[str, np.ndarray],
    source: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> ClipManifest:
    """Write ``clips/<recording_id>/<clip_name>/`` with media and ``clip.json``.

    Args:
        media_videos: ``camera_id -> (T, H, W, 3) uint8 RGB frames``; each is
            encoded to ``media/<camera_id>.mp4``. Camera order in the manifest
            is the insertion order of this mapping and defines the scene
            camera axis.
        overwrite: Refuse to touch an existing clip directory unless True.
    """
    validate_id_component(recording_id, field_name="recording_id")
    validate_id_component(clip_name, field_name="clip_name")
    if not media_videos:
        raise DatasetContractError("media_videos must contain at least one camera.")

    root = Path(dataset_root)
    clip_dir = root / CLIPS_DIR_NAME / recording_id / clip_name
    if clip_dir.exists() and not overwrite:
        raise DatasetContractError(
            f"clip directory already exists: {clip_dir}. Pass overwrite=True to replace it."
        )

    camera_ids = tuple(media_videos)
    for camera_id in camera_ids:
        validate_id_component(camera_id, field_name="camera_id")
    for camera_id, frames in media_videos.items():
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise DatasetContractError(
                f"media_videos[{camera_id!r}] must be (T, H, W, 3) uint8, "
                f"got shape={frames.shape} dtype={frames.dtype}."
            )
        if frames.shape[0] != num_frames or frames.shape[1:3] != (height, width):
            raise DatasetContractError(
                f"media_videos[{camera_id!r}] shape {frames.shape} disagrees with "
                f"num_frames={num_frames}, height={height}, width={width}."
            )

    media_dir = ensure_dir(clip_dir / MEDIA_DIR_NAME)
    video_paths: list[str] = []
    cameras: list[dict[str, Any]] = []
    for camera_id, frames in media_videos.items():
        out_path = media_dir / f"{camera_id}.mp4"
        save_video_rgb(np.asarray(frames), out_path, fps=float(fps))
        relative_path = f"{MEDIA_DIR_NAME}/{camera_id}.mp4"
        video_paths.append(relative_path)
        cameras.append(
            {
                "camera_id": camera_id,
                "video": relative_path,
                "source": source or {},
                "letterbox": None,
            }
        )

    payload: dict[str, Any] = {
        "version": CLIP_SCHEMA_VERSION,
        "clip_id": f"{recording_id}/{clip_name}",
        "recording_id": recording_id,
        "clip_name": clip_name,
        "fps": float(fps),
        "num_frames": int(num_frames),
        "width": int(width),
        "height": int(height),
        "camera_ids": list(camera_ids),
        "video_paths": video_paths,
        "cameras": cameras,
        "sync_source": "synthetic",
        "exported_at": utc_now_iso(),
    }
    save_json(payload, clip_dir / CLIP_MANIFEST_NAME)
    return ClipManifest.load(clip_dir)


def _scene_arrays_spec(scene: SceneResult) -> dict[str, dict[str, Any]]:
    spec: dict[str, dict[str, Any]] = {}
    for name in (
        "court_kp",
        "court_vis",
        "player_position",
        "player_yaw",
        "smpl_body_pose",
        "smpl_global_orient",
        "smpl_betas",
        "smpl_vertices_local",
        "ball_uv",
        "ball_vis",
        "ball_3d",
        "human_kp_2d",
        "human_kp_vis",
        "player_track_ids",
        "player_kp_3d",
    ):
        value = getattr(scene, name)
        if value is None:
            continue
        arr = np.asarray(value)
        spec[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return spec


def write_tennis_scene_annotation(
    manifest: ClipManifest,
    scene: SceneResult,
    *,
    generator: dict[str, Any] | None = None,
    pipeline_config_text: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``annotations/tennis_scene/`` for a clip; marker is written last.

    Returns the completion marker path.
    """
    ann_dir = tennis_scene_dir(manifest.clip_dir)
    marker_path = ann_dir / ANNOTATION_MARKER_NAME
    if marker_path.exists() and not overwrite:
        raise DatasetContractError(
            f"{manifest.clip_id}: completed tennis_scene annotation already exists "
            f"({marker_path}). Pass overwrite=True to regenerate."
        )
    ensure_dir(ann_dir)

    scene.save(ann_dir / SCENE_NPZ_NAME)
    if pipeline_config_text is not None:
        (ann_dir / "pipeline_config.yaml").write_text(
            pipeline_config_text, encoding="utf-8"
        )

    marker: dict[str, Any] = {
        "version": ANNOTATION_SCHEMA_VERSION,
        "clip_id": manifest.clip_id,
        "generator": generator or {},
        "generated_at": utc_now_iso(),
        "scene_result": SCENE_NPZ_NAME,
        "pipeline_config": "pipeline_config.yaml" if pipeline_config_text else None,
        "clip_manifest_sha256": manifest.digest(),
        "arrays": _scene_arrays_spec(scene),
    }
    save_json_atomic(marker, marker_path)
    return marker_path


def append_dataset_index(dataset_root: str | Path, manifest: ClipManifest) -> Path:
    """Append a clip to ``dataset.json``, creating the index on first use.

    Re-registering an existing ``clip_id`` is an explicit error; existing
    entries are never rewritten.
    """
    dataset = register_exported_clip(dataset_root, manifest.manifest_path)
    return Path(dataset.save(dataset_root))


__all__ = [
    "append_dataset_index",
    "write_clip_manifest",
    "write_tennis_scene_annotation",
]
