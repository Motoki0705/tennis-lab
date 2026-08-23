"""Scene IO helpers for BLCS visualization orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.visualization.io import BaseSceneBundle, resolve_cameras
from src.tasks.blcs.generate_dataset.io.dataset_io import load_scene
from src.utils.schema.court_normalization import CourtCoordinateNormalization


@dataclass(frozen=True)
class SceneBundle(BaseSceneBundle):
    """Loaded scene plus extracted artifacts for visualization."""

    scene: dict[str, Any] = field(default_factory=dict)
    gt_positions: np.ndarray = field(default_factory=lambda: np.empty(0))


def get_available_camera_indices(scene: dict[str, Any]) -> list[int]:
    """Return all camera indices available in the scene."""
    return list(range(int(scene["num_cameras"])))


def load_scene_bundle(
    scene_path: Path,
    camera: int,
    cameras: Sequence[int] | str | None,
    normalization: CourtCoordinateNormalization,
) -> SceneBundle:
    """Load scene and prepare GT/fps artifacts.

    Args:
        scene_path: Path to scene npz file.
        camera: Default single camera index when no camera list is selected.
        cameras: Optional explicit camera list.
        normalization: Runtime contract validated against root/scene metadata.

    Returns:
        SceneBundle containing scene object, GT positions, selected cameras and fps.
    """
    scene = load_scene(
        scene_path,
        court_coordinate_normalization=normalization,
    )
    gt_positions = scene["ball_pos_world"]
    if "fps_out" not in scene["meta"]:
        raise ValueError("BLCS scene meta.fps_out is required for visualization.")
    fps = float(scene["meta"]["fps_out"])

    selected_cameras = resolve_cameras(
        int(scene["num_cameras"]),
        camera,
        cameras,
        lambda: get_available_camera_indices(scene),
    )

    return SceneBundle(
        scene=scene,
        gt_positions=gt_positions,
        cameras=selected_cameras,
        fps=fps,
    )
