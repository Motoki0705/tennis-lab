"""Scene loading utilities for PLCS datasets.

This module provides functions to load scene data from npy + json directories.
It is separated from dataset_io.py to avoid circular imports when
used by dataset and visualization modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class AttrDict(dict[str, Any]):
    """Dict with attribute-style access for convenience."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_scene(filepath: str | Path) -> dict[str, Any]:
    """Load a scene from a npy + json scene directory.

    Args:
        filepath: Path to the scene directory.

    Returns:
        Dictionary with scene data including meta, position, rotation,
        canonical_pose_3d, num_cameras, and cameras list.
    """
    scene_dir = Path(filepath)

    with open(scene_dir / "meta.json") as f:
        meta = json.load(f)
    with open(scene_dir / "scalars.json") as f:
        scalars = json.load(f)

    num_cameras = int(scalars["num_cameras"])

    cameras = []
    for i in range(num_cameras):
        prefix = f"cam_{i}_"
        params = scalars[f"{prefix}params"]
        if isinstance(params, str):
            params = json.loads(params)
        cam_data = AttrDict(
            params=params,
            human_kp_uv=np.load(scene_dir / f"{prefix}human_kp_uv.npy"),
            human_kp_visible=np.load(scene_dir / f"{prefix}human_kp_visible.npy"),
            human_visibility_ratio=float(
                np.load(scene_dir / f"{prefix}human_visibility_ratio.npy")
            ),
            court_kp_uv=np.load(scene_dir / f"{prefix}court_kp_uv.npy"),
            court_kp_visible=np.load(scene_dir / f"{prefix}court_kp_visible.npy"),
            court_visibility_count=float(
                np.load(scene_dir / f"{prefix}court_visibility_count.npy")
            ),
        )
        cameras.append(cam_data)

    scene = AttrDict(
        meta=meta,
        position=np.load(scene_dir / "position.npy"),
        rotation=np.load(scene_dir / "rotation.npy"),
        canonical_pose_3d=np.load(scene_dir / "canonical_pose_3d.npy"),
        num_cameras=num_cameras,
        cameras=cameras,
        num_persons=int(scalars.get("num_persons", 1)),
    )

    # Include pre-computed COCO17 world joints when stored (AthletePose3D path).
    human_kp_3d_path = scene_dir / "human_kp_3d.npy"
    if human_kp_3d_path.exists():
        scene["human_kp_3d"] = np.load(human_kp_3d_path)

    person_present_path = scene_dir / "person_present.npy"
    if person_present_path.exists():
        scene["person_present"] = np.load(person_present_path)

    return scene
