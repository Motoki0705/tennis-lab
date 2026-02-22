"""Scene IO helpers for PLCS visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.plcs.generate_dataset.io.dataset_io import load_scene


@dataclass(frozen=True)
class SceneBundle:
    """Loaded scene and resolved runtime artifacts for visualization."""

    scene: Any
    cameras: list[int]
    fps: float


def _resolve_cameras(
    scene: Any,
    camera: int,
    cameras: list[int] | str | None,
) -> list[int]:
    num_cameras = int(scene.num_cameras)
    if cameras == "all":
        selected = list(range(num_cameras))
    elif cameras:
        selected = list(cameras)
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


def load_scene_bundle(
    scene_path: Path,
    camera: int,
    cameras: list[int] | str | None,
) -> SceneBundle:
    """Load scene and resolve selected cameras/fps for visualization."""
    scene = load_scene(scene_path)
    selected_cameras = _resolve_cameras(scene, camera=camera, cameras=cameras)
    fps = float(scene.meta.get("fps", 30.0))
    return SceneBundle(scene=scene, cameras=selected_cameras, fps=fps)
