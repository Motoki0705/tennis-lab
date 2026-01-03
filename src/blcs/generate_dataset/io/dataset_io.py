"""Dataset writer for BLCS dataset generation (PLCS-unified format).

Saves scene data in npz format with structure:
- meta: scene metadata (JSON)
- ball_pos_world: [T, 3] world coordinates
- ball_pos_norm: [T, 3] normalized coordinates
- ball_vel_world: [T, 3] velocities
- num_cameras: number of valid cameras
- cam_{i}_params: camera parameters for camera i
- cam_{i}_ball_uv: [T, 2] ball UV coordinates
- cam_{i}_ball_visible: [T] ball visibility
- cam_{i}_ball_visibility_ratio: visibility ratio
- cam_{i}_court_kp_uv: [20, 2] court keypoints UV
- cam_{i}_court_kp_visible: [20] court keypoint visibility
- cam_{i}_court_visibility_count: visible keypoint count
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.base.data.writer import BaseDatasetWriter
from src.blcs.data.types import (
    PYDANTIC_AVAILABLE,
    BLCSSceneMeta,
    BLCSSceneMetaModel,
    RallySceneMeta,
)
from src.blcs.generate_dataset.scene_generator import BLCSSceneData, RallySceneData

logger = logging.getLogger(__name__)


class BLCSDatasetWriter(BaseDatasetWriter):
    """Writes BLCS scene data to disk in npz format (PLCS-unified)."""
    scenes_dir: Path

    def __init__(self, output_dir: str | Path, validate: bool = False) -> None:
        """Initialize dataset writer.

        Args:
            output_dir: Output directory for dataset.
            validate: If True, use Pydantic models for runtime validation.

        """
        super().__init__(output_dir)
        self.validate = validate
        if validate and not PYDANTIC_AVAILABLE:
            logger.warning(
                "Pydantic validation requested but pydantic not installed. "
                "Install with: pip install pydantic>=2.10"
            )
            self.validate = False

    def save_scene(self, scene: BLCSSceneData) -> Path:
        """Save a single scene to npz file (1 scene = 1 file with N cameras).

        Args:
            scene: Scene data to save.

        Returns:
            Path: Path to saved file.

        """
        # Generate filename
        filename = f"{scene.scene_id}.npz"
        filepath = self.scenes_dir / filename

        # Create metadata using dataclass (with optional Pydantic validation)
        meta_dict = {
            "scene_id": scene.scene_id,
            "from_cell": scene.from_cell,
            "from_side": scene.from_side,
            "category": scene.category.value,
            "to_cell": scene.to_cell if scene.to_cell is not None else -1,
            "t_net": scene.t_net,
            "t_fence": scene.t_fence,
            "t_bounce1": scene.t_bounce1,
            "t_bounce2": scene.t_bounce2,
            "fps_out": scene.fps_out,
            "sim_fps": scene.sim_fps,
            "num_frames": scene.ball_pos_world.shape[0],
            "num_cameras_sampled": scene.num_cameras_sampled,
            "num_cameras": len(scene.cameras),
        }

        # Optionally validate with Pydantic (catches errors early)
        if self.validate and PYDANTIC_AVAILABLE:
            # This will raise ValidationError if data is invalid
            BLCSSceneMetaModel(**meta_dict)

        meta = BLCSSceneMeta(**meta_dict)

        # Build save dictionary
        save_dict = {
            # Metadata (as JSON string)
            "meta": json.dumps(meta.to_dict()),
            # 3D trajectory data (global)
            "ball_pos_world": scene.ball_pos_world.numpy(),
            "ball_pos_norm": scene.ball_pos_norm.numpy(),
            "ball_vel_world": scene.ball_vel_world.numpy(),
            # Number of cameras
            "num_cameras": np.array(len(scene.cameras)),
        }

        # Add per-camera data with cam_{i}_... keys
        camera_metas = []
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"

            # Camera parameters
            save_dict[f"{prefix}params"] = json.dumps(cam.camera_params)

            # Ball projections
            save_dict[f"{prefix}ball_uv"] = cam.ball_uv.astype(np.float32)
            save_dict[f"{prefix}ball_visible"] = cam.ball_visible.astype(bool)
            save_dict[f"{prefix}ball_visibility_ratio"] = np.array(
                cam.ball_visibility_ratio, dtype=np.float32
            )

            # Court keypoint projections
            save_dict[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
            save_dict[f"{prefix}court_kp_visible"] = cam.court_kp_visible.astype(bool)
            save_dict[f"{prefix}court_visibility_count"] = np.array(
                cam.court_visibility_count, dtype=np.float32
            )

            # Track camera meta for meta.json
            camera_metas.append(
                {
                    "ball_visibility_ratio": float(cam.ball_visibility_ratio),
                    "court_visibility_count": float(cam.court_visibility_count),
                }
            )

        # Save to npz
        np.savez_compressed(filepath, **save_dict)

        # Track for meta.json
        self.scene_records.append(
            {
                "file": filename,
                "scene_id": scene.scene_id,
                "category": scene.category.value,
                "num_frames": int(scene.ball_pos_world.shape[0]),
                "num_cameras_sampled": scene.num_cameras_sampled,
                "num_cameras": len(scene.cameras),
                "cameras": camera_metas,
            }
        )
        self.scene_counter += 1

        return filepath

    def save_rally_scene(self, scene: RallySceneData) -> Path:
        """Save a rally scene to npz file (1 rally = 1 file with N cameras).

        Args:
            scene: Rally scene data to save.

        Returns:
            Path: Path to saved file.

        """
        # Generate filename
        filename = f"{scene.scene_id}.npz"
        filepath = self.scenes_dir / filename

        # Create metadata
        meta = RallySceneMeta(
            scene_id=scene.scene_id,
            initial_from_cell=scene.initial_from_cell,
            initial_from_side=scene.initial_from_side,
            rally_length=scene.rally_length,
            end_reason=scene.end_reason,
            winner_side=scene.winner_side,
            shots=scene.shots,
            fps_out=scene.fps_out,
            sim_fps=scene.sim_fps,
            num_frames=scene.ball_pos_world.shape[0],
            num_cameras_sampled=scene.num_cameras_sampled,
            num_cameras=len(scene.cameras),
        )

        # Build save dictionary
        save_dict = {
            # Metadata (as JSON string)
            "meta": json.dumps(meta.to_dict()),
            # 3D trajectory data (global)
            "ball_pos_world": scene.ball_pos_world.numpy(),
            "ball_pos_norm": scene.ball_pos_norm.numpy(),
            "ball_vel_world": scene.ball_vel_world.numpy(),
            # Number of cameras
            "num_cameras": np.array(len(scene.cameras)),
            # Rally-specific data
            "rally_length": np.array(scene.rally_length),
            "end_reason": scene.end_reason,
        }

        # Add per-camera data with cam_{i}_... keys
        camera_metas = []
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"

            # Camera parameters
            save_dict[f"{prefix}params"] = json.dumps(cam.camera_params)

            # Ball projections
            save_dict[f"{prefix}ball_uv"] = cam.ball_uv.astype(np.float32)
            save_dict[f"{prefix}ball_visible"] = cam.ball_visible.astype(bool)
            save_dict[f"{prefix}ball_visibility_ratio"] = np.array(
                cam.ball_visibility_ratio, dtype=np.float32
            )

            # Court keypoint projections
            save_dict[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
            save_dict[f"{prefix}court_kp_visible"] = cam.court_kp_visible.astype(bool)
            save_dict[f"{prefix}court_visibility_count"] = np.array(
                cam.court_visibility_count, dtype=np.float32
            )

            # Track camera meta for meta.json
            camera_metas.append(
                {
                    "ball_visibility_ratio": float(cam.ball_visibility_ratio),
                    "court_visibility_count": float(cam.court_visibility_count),
                }
            )

        # Save to npz
        np.savez_compressed(filepath, **save_dict)

        # Track for meta.json
        self.scene_records.append(
            {
                "file": filename,
                "scene_id": scene.scene_id,
                "rally_length": scene.rally_length,
                "end_reason": scene.end_reason,
                "winner_side": scene.winner_side,
                "num_frames": int(scene.ball_pos_world.shape[0]),
                "num_cameras_sampled": scene.num_cameras_sampled,
                "num_cameras": len(scene.cameras),
                "cameras": camera_metas,
            }
        )
        self.scene_counter += 1

        return filepath
def load_scene(filepath: str | Path) -> dict:
    """Load a scene from npz file (PLCS-unified format).

    Args:
        filepath: Path to npz file.

    Returns:
        dict: Scene data with:
            - meta: parsed metadata
            - ball_pos_world, ball_pos_norm, ball_vel_world: 3D data
            - num_cameras: number of cameras
            - cameras: list of camera data dicts

    """
    data = np.load(filepath, allow_pickle=True)

    # Parse metadata
    meta = json.loads(str(data["meta"]))
    num_cameras = int(data["num_cameras"])

    # Load camera data
    cameras = []
    for i in range(num_cameras):
        prefix = f"cam_{i}_"
        cam_data = {
            "params": json.loads(str(data[f"{prefix}params"])),
            "ball_uv": data[f"{prefix}ball_uv"],
            "ball_visible": data[f"{prefix}ball_visible"],
            "ball_visibility_ratio": float(data[f"{prefix}ball_visibility_ratio"]),
            "court_kp_uv": data[f"{prefix}court_kp_uv"],
            "court_kp_visible": data[f"{prefix}court_kp_visible"],
            "court_visibility_count": float(data[f"{prefix}court_visibility_count"]),
        }
        cameras.append(cam_data)

    return {
        "meta": meta,
        "ball_pos_world": data["ball_pos_world"],
        "ball_pos_norm": data["ball_pos_norm"],
        "ball_vel_world": data["ball_vel_world"],
        "num_cameras": num_cameras,
        "cameras": cameras,
    }
