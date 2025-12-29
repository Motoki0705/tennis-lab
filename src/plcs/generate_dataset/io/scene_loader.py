"""Scene loading utilities for PLCS datasets.

This module provides functions to load scene data from npz files.
It is separated from dataset_io.py to avoid circular imports when
used by src.plcs.data.dataset and src.plcs.data.sequence_dataset.
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
    """Load a scene from npz file (PLCS-unified format).

    Args:
        filepath: Path to the npz file.

    Returns:
        Dictionary with scene data including meta, position, rotation,
        canonical_pose_3d, num_cameras, and cameras list.

    """
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
