"""Scene IO helpers for BLCS visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.blcs.generate_dataset.io.dataset_io import load_scene


@dataclass(frozen=True)
class SceneBundle:
    """Loaded scene plus extracted artifacts for visualization."""

    scene: dict[str, Any]
    gt_positions: np.ndarray
    cameras: list[int]
    fps: float


def _resolve_cameras(
    scene: dict[str, Any],
    camera: int,
    cameras: list[int] | str | None,
) -> list[int]:
    """Resolve and validate camera indices."""
    num_cameras = int(scene["num_cameras"])
    if cameras == "all":
        selected = get_available_camera_indices(scene)
    elif cameras:
        selected = cameras
    else:
        selected = [camera]

    if not selected:
        raise ValueError("No cameras selected.")

    for cam_idx in selected:
        if cam_idx < 0 or cam_idx >= num_cameras:
            raise ValueError(
                f"Camera {cam_idx} out of range (0-{num_cameras - 1})."
            )
    return selected


def get_available_camera_indices(scene: dict[str, Any]) -> list[int]:
    """Return all camera indices available in the scene."""
    return list(range(int(scene["num_cameras"])))


def load_scene_bundle(
    scene_path: Path,
    camera: int,
    cameras: list[int] | str | None,
) -> SceneBundle:
    """Load scene and prepare GT/fps artifacts.

    Args:
        scene_path: Path to scene npz file.
        camera: Fallback single camera index.
        cameras: Optional explicit camera list.

    Returns:
        SceneBundle containing scene object, GT positions, selected cameras and fps.
    """
    scene = load_scene(scene_path)
    gt_positions = scene["ball_pos_world"]
    fps = float(scene["meta"].get("fps_out", 30.0))

    selected_cameras = _resolve_cameras(scene, camera=camera, cameras=cameras)

    return SceneBundle(
        scene=scene,
        gt_positions=gt_positions,
        cameras=selected_cameras,
        fps=fps,
    )
