"""Compressed NPZ archive I/O for :class:`src.tennis_scene.schema.SceneResult`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from src.tennis_scene.schema import SceneResult


def _metadata_sidecar_path(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def _optional_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    dtype: DTypeLike = None,
) -> NDArray[Any] | None:
    if name not in archive.files:
        return None
    return np.asarray(archive[name], dtype=dtype)


def save_scene_result(result: SceneResult, path: str | Path) -> None:
    """Save a scene result to compressed NPZ plus its mandatory JSON sidecar."""
    archive_path = Path(path)
    if archive_path.suffix != ".npz":
        raise ValueError(
            f"Scene archive path must use the .npz suffix: {archive_path}"
        )
    if not isinstance(result.metadata, dict):
        raise TypeError("Scene metadata must be a dictionary")
    metadata_text = json.dumps(
        result.metadata,
        ensure_ascii=False,
        indent=2,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "num_frames": result.num_frames,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "court_kp": result.court_kp,
        "court_vis": result.court_vis,
        "player_position": result.player_position,
        "player_yaw": result.player_yaw,
        "smpl_body_pose": result.smpl_body_pose,
        "smpl_global_orient": result.smpl_global_orient,
        "smpl_betas": result.smpl_betas,
    }
    optional_arrays = {
        "smpl_vertices_local": result.smpl_vertices_local,
        "ball_uv": result.ball_uv,
        "ball_vis": result.ball_vis,
        "ball_3d": result.ball_3d,
        "human_kp_2d": result.human_kp_2d,
        "human_kp_vis": result.human_kp_vis,
        "player_track_ids": result.player_track_ids,
        "player_kp_3d": result.player_kp_3d,
    }
    arrays.update(
        (name, value) for name, value in optional_arrays.items() if value is not None
    )

    np.savez_compressed(archive_path, **arrays)
    with _metadata_sidecar_path(archive_path).open("w", encoding="utf-8") as handle:
        handle.write(metadata_text)


def load_scene_result(path: str | Path) -> SceneResult:
    """Load a scene archive, rejecting archives without object metadata."""
    archive_path = Path(path)
    sidecar_path = _metadata_sidecar_path(archive_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Scene metadata sidecar not found: {sidecar_path}")
    with sidecar_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError(f"Scene metadata must be a JSON object: {sidecar_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        return SceneResult(
            num_frames=int(archive["num_frames"]),
            fps=float(archive["fps"]),
            width=int(archive["width"]),
            height=int(archive["height"]),
            court_kp=np.asarray(archive["court_kp"], dtype=np.float32),
            court_vis=np.asarray(archive["court_vis"], dtype=np.float32),
            player_position=np.asarray(archive["player_position"], dtype=np.float32),
            player_yaw=np.asarray(archive["player_yaw"], dtype=np.float32),
            smpl_body_pose=np.asarray(archive["smpl_body_pose"], dtype=np.float32),
            smpl_global_orient=np.asarray(
                archive["smpl_global_orient"], dtype=np.float32
            ),
            smpl_betas=np.asarray(archive["smpl_betas"], dtype=np.float32),
            smpl_vertices_local=_optional_array(
                archive, "smpl_vertices_local", dtype=np.float32
            ),
            ball_uv=_optional_array(archive, "ball_uv", dtype=np.float32),
            ball_vis=_optional_array(archive, "ball_vis", dtype=np.bool_),
            ball_3d=_optional_array(archive, "ball_3d", dtype=np.float32),
            human_kp_2d=_optional_array(archive, "human_kp_2d", dtype=np.float32),
            human_kp_vis=_optional_array(
                archive, "human_kp_vis", dtype=np.float32
            ),
            player_track_ids=_optional_array(
                archive, "player_track_ids", dtype=np.int32
            ),
            player_kp_3d=_optional_array(
                archive, "player_kp_3d", dtype=np.float32
            ),
            metadata=metadata,
        )
