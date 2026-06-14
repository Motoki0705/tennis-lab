"""Scene IO helpers for PLCS visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tasks.base.visualization.io import BaseSceneBundle, resolve_cameras
from src.tasks.plcs.generate_dataset.io.dataset_io import load_scene


@dataclass(frozen=True)
class SceneBundle(BaseSceneBundle):
    """Loaded scene and resolved runtime artifacts for visualization."""

    scene: Any = None


def load_scene_bundle(
    scene_path: Path,
    camera: int,
    cameras: list[int] | str | None,
) -> SceneBundle:
    """Load scene and resolve selected cameras/fps for visualization."""
    scene = load_scene(scene_path)
    num_cameras = int(scene.num_cameras)
    selected_cameras = resolve_cameras(
        num_cameras,
        camera,
        cameras,
        lambda: list(range(num_cameras)),
    )
    fps = float(scene.meta.get("fps", 30.0))
    return SceneBundle(scene=scene, cameras=selected_cameras, fps=fps)
