"""Scene IO helpers for PLCS visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.tasks.base.visualization.io import BaseSceneBundle, resolve_cameras
from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
from src.utils.schema.court_normalization import CourtCoordinateNormalization


@dataclass(frozen=True)
class SceneBundle(BaseSceneBundle):
    """Loaded scene and resolved runtime artifacts for visualization."""

    scene: Any = None


def load_scene_bundle(
    scene_path: Path,
    camera: int,
    cameras: tuple[int, ...] | Literal["all"] | None,
    *,
    court_coordinate_normalization: CourtCoordinateNormalization,
) -> SceneBundle:
    """Validate and load a scene, then resolve selected cameras and fps."""
    scene: Any = load_scene(
        scene_path,
        court_coordinate_normalization=court_coordinate_normalization,
    )
    num_cameras = int(scene.num_cameras)
    selected_cameras = resolve_cameras(
        num_cameras,
        camera,
        cameras,
        lambda: list(range(num_cameras)),
    )
    fps = float(scene.meta["fps"])
    if fps <= 0.0:
        raise ValueError("PLCS scene meta.fps must be positive.")
    return SceneBundle(scene=scene, cameras=selected_cameras, fps=fps)
