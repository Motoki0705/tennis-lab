"""Dataset writer for BLCS dataset generation.

Saves scene data as a directory per scene with structure:
- meta.json: scene metadata
- scalars.json: scalar values (num_cameras, rally_length, end_reason,
  camera parameters)
- {key}.npy: array data files (ball_pos_world, ball_pos_norm, etc.)

Each camera produces per-camera npy files (cam_{i}_ball_uv.npy, etc.)
and its parameters are stored in scalars.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.data.dataset_writer import BaseDatasetWriter
from src.tasks.blcs.data.types import (
    BLCSSceneMeta,
)
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData

logger = logging.getLogger(__name__)


class BLCSDatasetWriter(BaseDatasetWriter):
    """Writes BLCS scene data to disk as npy + json directories."""

    scenes_dir: Path

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__(output_dir)

    def _build_scene_meta(self, scene: BLCSSceneData) -> BLCSSceneMeta:
        scene_meta_dict: dict[str, Any] = {
            "scene_id": scene.scene_id,
            "initial_from_cell": scene.initial_from_cell,
            "initial_from_side": scene.initial_from_side,
            "rally_length": scene.rally_length,
            "end_reason": scene.end_reason,
            "winner_side": scene.winner_side,
            "shots": scene.shots,
            "fps_out": scene.fps_out,
            "sim_fps": scene.sim_fps,
            "num_frames": int(scene.ball_pos_world.shape[0]),
            "num_cameras_sampled": scene.num_cameras_sampled,
            "num_cameras": len(scene.cameras),
            "physics_config": scene.physics_config_dict,
            "court_config": scene.court_config_dict,
            "track_instances": scene.track_instances,
        }

        return BLCSSceneMeta(**scene_meta_dict)

    def _append_camera_arrays(
        self,
        arrays: dict[str, np.ndarray],
        scalars: dict[str, Any],
        scene: BLCSSceneData,
    ) -> list[dict[str, float]]:
        camera_records: list[dict[str, float]] = []
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"
            scalars[f"{prefix}params"] = cam.camera_params
            arrays[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
            arrays[f"{prefix}court_kp_vis"] = cam.court_kp_vis.astype(bool)
            arrays[f"{prefix}court_visibility_count"] = np.array(
                cam.court_visibility_count,
                dtype=np.float32,
            )
            # Ball-specific per-camera arrays (not shared with PLCS).
            arrays[f"{prefix}ball_uv"] = cam.ball_uv.astype(np.float32)
            arrays[f"{prefix}ball_vis"] = cam.ball_vis.astype(bool)
            arrays[f"{prefix}ball_visibility_ratio"] = np.array(
                cam.ball_visibility_ratio,
                dtype=np.float32,
            )
            camera_records.append(
                {
                    "ball_visibility_ratio": float(cam.ball_visibility_ratio),
                    "court_visibility_count": float(cam.court_visibility_count),
                }
            )
        return camera_records

    def save_scene(self, scene: BLCSSceneData) -> Path:
        """Save a BLCS scene (rally) as a directory with npy + json files."""
        dirname = scene.scene_id
        scene_path: Path = self.scenes_dir / dirname
        scene_path.mkdir(parents=True, exist_ok=True)
        scene_meta = self._build_scene_meta(scene)

        arrays: dict[str, np.ndarray] = {
            "ball_pos_world": scene.ball_pos_world.numpy(),
            "ball_pos_norm": scene.ball_pos_norm.numpy(),
            "ball_vel_world": scene.ball_vel_world.numpy(),
        }
        if scene.ball_present is not None:
            arrays["ball_present"] = scene.ball_present.cpu().numpy()
        scalars: dict[str, Any] = {
            "num_cameras": len(scene.cameras),
            "num_balls": scene.num_balls,
            "rally_length": scene.rally_length,
            "end_reason": scene.end_reason,
        }
        camera_records = self._append_camera_arrays(arrays, scalars, scene)

        self._write_scene_files(scene_path, scene_meta, scalars, arrays)

        self.scene_records.append(
            {
                "file": dirname,
                "scene_id": scene.scene_id,
                "rally_length": scene.rally_length,
                "end_reason": scene.end_reason,
                "winner_side": scene.winner_side,
                "num_frames": int(scene.ball_pos_world.shape[0]),
                "num_cameras_sampled": scene.num_cameras_sampled,
                "num_cameras": len(scene.cameras),
                "cameras": camera_records,
            }
        )
        self.scene_counter += 1
        return scene_path


def load_scene(filepath: str | Path) -> dict:
    """Load a scene from a npy + json scene directory.

    Args:
        filepath: Path to the scene directory.

    Returns:
        dict: Scene data with:
            - meta: parsed metadata
            - ball_pos_world, ball_pos_norm, ball_vel_world: 3D data
            - num_cameras: number of cameras
            - cameras: list of camera data dicts
    """
    scene_dir = Path(filepath)

    with open(scene_dir / "meta.json") as f:
        scene_meta = json.load(f)
    with open(scene_dir / "scalars.json") as f:
        scalars = json.load(f)

    num_cameras = int(scalars["num_cameras"])

    cameras = []
    for i in range(num_cameras):
        prefix = f"cam_{i}_"
        params = scalars[f"{prefix}params"]
        if isinstance(params, str):
            params = json.loads(params)
        cam_data = {
            "params": params,
            "ball_uv": np.load(scene_dir / f"{prefix}ball_uv.npy"),
            "ball_vis": np.load(scene_dir / f"{prefix}ball_vis.npy"),
            "ball_visibility_ratio": float(
                np.load(scene_dir / f"{prefix}ball_visibility_ratio.npy")
            ),
            "court_kp_uv": np.load(scene_dir / f"{prefix}court_kp_uv.npy"),
            "court_kp_vis": np.load(scene_dir / f"{prefix}court_kp_vis.npy"),
            "court_visibility_count": float(
                np.load(scene_dir / f"{prefix}court_visibility_count.npy")
            ),
        }
        cameras.append(cam_data)

    result = {
        "meta": scene_meta,
        "ball_pos_world": np.load(scene_dir / "ball_pos_world.npy"),
        "ball_pos_norm": np.load(scene_dir / "ball_pos_norm.npy"),
        "ball_vel_world": np.load(scene_dir / "ball_vel_world.npy"),
        "num_cameras": num_cameras,
        "cameras": cameras,
    }
    ball_present_path = scene_dir / "ball_present.npy"
    if ball_present_path.exists():
        result["ball_present"] = np.load(ball_present_path)
    if "num_balls" not in scalars:
        raise ValueError(
            "BLCS scene is incompatible: required scalar num_balls is missing."
        )
    result["num_balls"] = int(scalars["num_balls"])
    return result
