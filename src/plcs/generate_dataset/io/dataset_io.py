"""Dataset I/O utilities for PLCS dataset generation (PLCS-unified format)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from src.base.data.writer import BaseDatasetWriter
from src.plcs.data.types import (
    PYDANTIC_AVAILABLE,
    PLCSSceneMeta,
    PLCSSceneMetaModel,
)
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


class PLCSDatasetWriter(BaseDatasetWriter):
    """Writes PLCS scene data to disk in npz format (PLCS-unified)."""
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

    def save_scene(self, scene: SceneData) -> Path:
        """Save a single scene to npz file (1 scene = 1 file with N cameras).

        Args:
            scene: Scene data to save.

        Returns:
            Path: Path to saved file.

        """
        filename = f"{scene.meta['scene_id']}.npz"
        filepath = self.scenes_dir / filename

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

        # Optionally validate with Pydantic (catches errors early)
        if self.validate and PYDANTIC_AVAILABLE:
            # This will raise ValidationError if data is invalid
            PLCSSceneMetaModel(**meta_dict)

        meta = PLCSSceneMeta(**meta_dict)

        save_dict: dict[str, SavezValue] = {
            "meta": json.dumps(meta.to_dict()),
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
