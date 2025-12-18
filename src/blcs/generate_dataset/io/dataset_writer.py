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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.blcs.generate_dataset.scene_generator import BLCSSceneData

logger = logging.getLogger(__name__)


class BLCSDatasetWriter:
    """Writes BLCS scene data to disk in npz format (PLCS-unified)."""

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize dataset writer.

        Args:
            output_dir: Output directory for dataset.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.scenes_dir = self.output_dir / "scenes"
        self.scenes_dir.mkdir(exist_ok=True)

        # Track scenes for meta.json
        self.scene_records: list[dict] = []
        self.scene_counter = 0

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

        # Prepare metadata
        meta = {
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

        # Build save dictionary
        save_dict = {
            # Metadata (as JSON string)
            "meta": json.dumps(meta),
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

    def save_split_info(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Save train/val/test split information.

        Each scene is one file, so splits are straightforward.

        Args:
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed for reproducibility.

        """
        import random

        # Get all scene files
        scene_files = [r["file"] for r in self.scene_records]

        # Shuffle
        random.seed(seed)
        random.shuffle(scene_files)

        # Split
        n_total = len(scene_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": scene_files[:n_train],
            "val": scene_files[n_train : n_train + n_val],
            "test": scene_files[n_train + n_val :],
        }

        # Save split files
        for split_name, filenames in splits.items():
            split_file = self.output_dir / f"{split_name}.txt"
            with open(split_file, "w") as f:
                for filename in filenames:
                    f.write(f"{filename}\n")

            logger.info(f"Saved {split_name} split: {len(filenames)} scenes")

        # Save full split info as JSON
        split_info = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "n_scenes": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
            },
        }

        with open(self.output_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

    def save_meta_json(self, config: dict | None = None) -> None:
        """Save meta.json with all scene information.

        Args:
            config: Generator configuration (optional).

        """
        # Calculate statistics
        total_cameras = sum(r["num_cameras"] for r in self.scene_records)
        avg_cameras = (
            total_cameras / len(self.scene_records) if self.scene_records else 0
        )

        meta = {
            "generated_at": datetime.now().isoformat(),
            "config": config or {},
            "stats": {
                "total_scenes": len(self.scene_records),
                "total_cameras": total_cameras,
                "avg_cameras_per_scene": avg_cameras,
            },
            "scenes": self.scene_records,
        }

        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"meta.json saved: {len(self.scene_records)} scenes, "
            f"{total_cameras} cameras"
        )

    def save_dataset_info(self, stats: dict) -> None:
        """Save dataset statistics and metadata.

        Args:
            stats: Statistics from generator.

        """
        info = {
            "total_scenes": len(self.scene_records),
            "total_cameras": stats.get("total_cameras", 0),
            "avg_cameras_per_scene": stats.get("avg_cameras_per_scene", 0),
            "camera_acceptance_rate": stats.get("camera_acceptance_rate", 0),
            "category_distribution": stats.get("category_counts", {}),
        }

        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Dataset info saved: {len(self.scene_records)} scenes")


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
