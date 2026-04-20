"""Dataset I/O utilities for PLCS dataset generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from src.tasks.base.data.dataset_writer import BaseDatasetWriter
from src.tasks.plcs.data.types import PLCSSceneMeta

# Re-export load_scene for backwards compatibility
from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene as load_scene
from src.tasks.plcs.generate_dataset.scene_generator import SceneData

logger = logging.getLogger(__name__)


class PLCSDatasetWriter(BaseDatasetWriter):
    """Writes PLCS scene data to disk as npy + json directories."""
    scenes_dir: Path

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__(output_dir)

    def save_scene(self, scene: SceneData) -> Path:
        """Save a single scene as a directory with npy + json files.

        Args:
            scene: Scene data to save.

        Returns:
            Path: Path to saved scene directory.
        """
        dirname = scene.meta["scene_id"]
        scene_path = self.scenes_dir / dirname
        scene_path.mkdir(parents=True, exist_ok=True)

        # Create metadata using dataclass (with optional Pydantic validation)
        meta_dict = {
            "scene_id": scene.meta["scene_id"],
            "motion_source": scene.meta["motion_source"],
            "motion_category": scene.meta["motion_category"],
            "gender": scene.meta["gender"],
            "fps": scene.meta["fps"],
            "num_frames": scene.meta["num_frames"],
            "initial_position": scene.meta["initial_position"],
            "initial_yaw": scene.meta["initial_yaw"],
            "num_cameras_sampled": scene.meta["num_cameras_sampled"],
            "num_cameras": len(scene.cameras),
        }

        meta = PLCSSceneMeta(**meta_dict)

        arrays: dict[str, np.ndarray] = {
            "position": np.asarray(scene.position),
            "rotation": np.asarray(scene.rotation),
            "canonical_pose_3d": np.asarray(scene.canonical_pose_3d),
        }

        scalars: dict[str, Any] = {
            "num_cameras": len(scene.cameras),
        }

        # Store pre-computed COCO17 world joints when available
        if scene.human_kp_3d is not None:
            arrays["human_kp_3d"] = np.asarray(scene.human_kp_3d).astype(
                np.float32
            )

        camera_metas = []
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"
            scalars[f"{prefix}params"] = cam.camera_params
            arrays[f"{prefix}human_kp_uv"] = cam.human_kp_uv.astype(np.float32)
            arrays[f"{prefix}human_kp_visible"] = cam.human_kp_visible.astype(bool)
            arrays[f"{prefix}human_visibility_ratio"] = np.array(
                cam.human_visibility_ratio, dtype=np.float32
            )
            arrays[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
            arrays[f"{prefix}court_kp_visible"] = cam.court_kp_visible.astype(bool)
            arrays[f"{prefix}court_visibility_count"] = np.array(
                cam.court_visibility_count, dtype=np.float32
            )

            camera_metas.append(
                {
                    "human_visibility_ratio": float(cam.human_visibility_ratio),
                    "court_visibility_count": float(cam.court_visibility_count),
                }
            )

        # Write meta.json
        with open(scene_path / "meta.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        # Write scalars.json
        with open(scene_path / "scalars.json", "w") as f:
            json.dump(scalars, f, indent=2)

        # Write array files
        for key, arr in arrays.items():
            np.save(scene_path / f"{key}.npy", arr)

        self.scene_records.append(
            {
                "file": dirname,
                "scene_id": scene.meta["scene_id"],
                "motion_category": scene.meta["motion_category"],
                "num_frames": int(scene.meta["num_frames"]),
                "num_cameras_sampled": scene.meta["num_cameras_sampled"],
                "num_cameras": len(scene.cameras),
                "cameras": camera_metas,
            }
        )
        self.scene_counter += 1

        return scene_path
