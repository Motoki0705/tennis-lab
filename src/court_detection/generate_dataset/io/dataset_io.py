"""Dataset I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_scene(
    path: Path,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    camera_params: dict[str, Any],
    image: np.ndarray | None = None,
) -> None:
    """Save a scene to disk.

    Args:
        path: Output path (npz file).
        keypoints: 2D keypoint coordinates (K, 2).
        visibility: Visibility flags (K,).
        camera_params: Camera parameters dict.
        image: Optional rendered image.
    """
    data = {
        "keypoints": keypoints,
        "visibility": visibility,
        "camera_center": camera_params.get("center", [0, 0, 0]),
        "camera_look_at": camera_params.get("look_at", [0, 0, 0]),
        "camera_hfov": camera_params.get("hfov", 60.0),
        "image_size": camera_params.get("image_size", [1280, 720]),
    }

    if image is not None:
        data["image"] = image

    np.savez(path, **data)


def load_scene(path: Path) -> dict[str, Any]:
    """Load a scene from disk.

    Args:
        path: Path to npz file.

    Returns:
        Scene dictionary.
    """
    data = np.load(path)

    scene = {
        "keypoints": data["keypoints"],
        "visibility": data["visibility"],
        "camera_params": {
            "center": data["camera_center"].tolist() if "camera_center" in data else None,
            "look_at": data["camera_look_at"].tolist() if "camera_look_at" in data else None,
            "hfov": float(data["camera_hfov"]) if "camera_hfov" in data else None,
            "image_size": data["image_size"].tolist() if "image_size" in data else None,
        },
    }

    if "image" in data:
        scene["image"] = data["image"]

    return scene
