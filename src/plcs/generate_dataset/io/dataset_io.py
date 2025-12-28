"""Dataset I/O utilities for PLCS dataset generation (PLCS-unified format)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.plcs.generate_dataset.scene_generator import SceneData

# Type alias for values accepted by np.savez_compressed
SavezValue: TypeAlias = npt.ArrayLike | bool | int | float | complex | str | bytes

logger = logging.getLogger(__name__)


class AttrDict(dict[str, Any]):
    """Dict with attribute-style access for convenience."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class PLCSDatasetWriter:
    """Writes PLCS scene data to disk in npz format (PLCS-unified)."""

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

    def save_scene(self, scene: SceneData) -> Path:
        """Save a single scene to npz file (1 scene = 1 file with N cameras).

        Args:
            scene: Scene data to save.

        Returns:
            Path: Path to saved file.

        """
        filename = f"{scene.meta['scene_id']}.npz"
        filepath = self.scenes_dir / filename

        meta = {
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

        save_dict: dict[str, SavezValue] = {
            "meta": json.dumps(meta),
            "position": np.asarray(scene.position),
            "rotation": np.asarray(scene.rotation),
            "canonical_pose_3d": np.asarray(scene.canonical_pose_3d),
            "num_cameras": np.array(len(scene.cameras)),
        }

        camera_metas = []
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"
            save_dict[f"{prefix}params"] = json.dumps(cam.camera_params)
            save_dict[f"{prefix}human_kp_uv"] = cam.human_kp_uv.astype(np.float32)
            save_dict[f"{prefix}human_kp_visible"] = cam.human_kp_visible.astype(bool)
            save_dict[f"{prefix}human_visibility_ratio"] = np.array(
                cam.human_visibility_ratio, dtype=np.float32
            )
            save_dict[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
            save_dict[f"{prefix}court_kp_visible"] = cam.court_kp_visible.astype(bool)
            save_dict[f"{prefix}court_visibility_count"] = np.array(
                cam.court_visibility_count, dtype=np.float32
            )

            camera_metas.append(
                {
                    "human_visibility_ratio": float(cam.human_visibility_ratio),
                    "court_visibility_count": float(cam.court_visibility_count),
                }
            )

        np.savez_compressed(filepath, **cast(Any, save_dict))

        self.scene_records.append(
            {
                "file": filename,
                "scene_id": scene.meta["scene_id"],
                "motion_category": scene.meta["motion_category"],
                "num_frames": int(scene.meta["num_frames"]),
                "num_cameras_sampled": scene.meta["num_cameras_sampled"],
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

        Args:
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed for reproducibility.

        """
        import random

        scene_files = [r["file"] for r in self.scene_records]

        random.seed(seed)
        random.shuffle(scene_files)

        n_total = len(scene_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": scene_files[:n_train],
            "val": scene_files[n_train : n_train + n_val],
            "test": scene_files[n_train + n_val :],
        }

        for split_name, filenames in splits.items():
            split_file = self.output_dir / f"{split_name}.txt"
            with open(split_file, "w") as f:
                for filename in filenames:
                    f.write(f"{filename}\n")

            logger.info("Saved %s split: %s scenes", split_name, len(filenames))

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
            "meta.json saved: %s scenes, %s cameras",
            len(self.scene_records),
            total_cameras,
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
            "human_visibility_threshold": stats.get("human_visibility_threshold", 0),
            "court_visibility_threshold": stats.get("court_visibility_threshold", 0),
        }

        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        logger.info("Dataset info saved: %s scenes", len(self.scene_records))


def load_scene(filepath: str | Path) -> dict:
    """Load a scene from npz file (PLCS-unified format)."""
    data = np.load(filepath, allow_pickle=True)

    meta_raw = data["meta"].item()
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    num_cameras = int(data["num_cameras"])

    cameras = []
    for i in range(num_cameras):
        prefix = f"cam_{i}_"
        params_raw = data[f"{prefix}params"].item()
        if isinstance(params_raw, (bytes, bytearray)):
            params_raw = params_raw.decode("utf-8")
        params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
        cam_data = AttrDict(
            params=params,
            human_kp_uv=data[f"{prefix}human_kp_uv"],
            human_kp_visible=data[f"{prefix}human_kp_visible"],
            human_visibility_ratio=float(data[f"{prefix}human_visibility_ratio"]),
            court_kp_uv=data[f"{prefix}court_kp_uv"],
            court_kp_visible=data[f"{prefix}court_kp_visible"],
            court_visibility_count=float(data[f"{prefix}court_visibility_count"]),
        )
        cameras.append(cam_data)

    return AttrDict(
        meta=meta,
        position=data["position"],
        rotation=data["rotation"],
        canonical_pose_3d=data["canonical_pose_3d"],
        num_cameras=num_cameras,
        cameras=cameras,
    )
