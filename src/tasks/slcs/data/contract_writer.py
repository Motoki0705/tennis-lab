"""Reference writer for the issue #634 dataset contract.

This is the executable counterpart of :mod:`src.tasks.slcs.data.contract`.
It exists so that

- tests and smoke runs can materialize a contract-conformant dataset, and
- small real datasets can be assembled manually until the issue #634
  ``generate_dataset`` pipeline lands.

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
    CLIP_FORMAT_VERSION,
    CLIP_MANIFEST_NAME,
    CLIPS_DIR_NAME,
    DATASET_FORMAT_VERSION,
    DATASET_INDEX_NAME,
    MEDIA_DIR_NAME,
    SCENE_NPZ_NAME,
    TENNIS_SCENE_ANNOTATION_KIND,
    TENNIS_SCENE_ANNOTATION_VERSION,
    ClipManifest,
    DatasetContractError,
    tennis_scene_dir,
    validate_id_component,
)
from src.tennis_scene.io import SceneResult
from src.utils.io import ensure_dir, load_json, save_json, save_json_atomic, utc_now_iso
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
    media: dict[str, str] = {}
    for camera_id, frames in media_videos.items():
        out_path = media_dir / f"{camera_id}.mp4"
        save_video_rgb(np.asarray(frames), out_path, fps=float(fps))
        media[camera_id] = f"{MEDIA_DIR_NAME}/{camera_id}.mp4"

    payload: dict[str, Any] = {
        "format_version": CLIP_FORMAT_VERSION,
        "clip_id": f"{recording_id}/{clip_name}",
        "recording_id": recording_id,
        "clip_name": clip_name,
        "fps": float(fps),
        "num_frames": int(num_frames),
        "width": int(width),
        "height": int(height),
        "camera_ids": list(camera_ids),
        "media": media,
        "source": source or {},
        "created_at": utc_now_iso(),
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
        (ann_dir / "pipeline_config.yaml").write_text(pipeline_config_text, encoding="utf-8")

    marker: dict[str, Any] = {
        "format_version": TENNIS_SCENE_ANNOTATION_VERSION,
        "kind": TENNIS_SCENE_ANNOTATION_KIND,
        "created_at": utc_now_iso(),
        "generator": generator or {},
        "arrays": _scene_arrays_spec(scene),
        "input_manifest_digest": manifest.digest(),
    }
    save_json_atomic(marker, marker_path)
    return marker_path


def append_dataset_index(dataset_root: str | Path, manifest: ClipManifest) -> Path:
    """Append a clip to ``dataset.json``, creating the index on first use.

    Re-registering an existing ``clip_id`` is an explicit error; existing
    entries are never rewritten.
    """
    root = Path(dataset_root)
    index_path = root / DATASET_INDEX_NAME
    if index_path.exists():
        payload = load_json(index_path)
        if not isinstance(payload, dict):
            raise DatasetContractError(f"{index_path} must contain a JSON object.")
        if payload.get("format_version") != DATASET_FORMAT_VERSION:
            raise DatasetContractError(
                f"{index_path} has format_version={payload.get('format_version')!r}; "
                f"writer supports {DATASET_FORMAT_VERSION}."
            )
        clips = payload.get("clips")
        if not isinstance(clips, list):
            raise DatasetContractError(f"{index_path} must contain a 'clips' list.")
    else:
        ensure_dir(root)
        payload = {
            "format_version": DATASET_FORMAT_VERSION,
            "created_at": utc_now_iso(),
            "clips": [],
        }
        clips = payload["clips"]

    if any(entry.get("clip_id") == manifest.clip_id for entry in clips):
        raise DatasetContractError(
            f"{index_path}: clip_id {manifest.clip_id!r} is already registered."
        )
    clips.append(
        {
            "clip_id": manifest.clip_id,
            "recording_id": manifest.recording_id,
            "clip_name": manifest.clip_name,
            "path": f"{CLIPS_DIR_NAME}/{manifest.recording_id}/{manifest.clip_name}",
        }
    )
    payload["updated_at"] = utc_now_iso()
    save_json_atomic(payload, index_path)
    return index_path


__all__ = [
    "append_dataset_index",
    "write_clip_manifest",
    "write_tennis_scene_annotation",
]
