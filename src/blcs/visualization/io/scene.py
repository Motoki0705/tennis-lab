"""Scene IO helpers for BLCS visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.blcs.generate_dataset.io.dataset_io import load_scene
from src.blcs.visualization.api.predict import PredictorInputs


@dataclass(frozen=True)
class SceneBundle:
    """Loaded scene plus extracted artifacts for visualization."""

    scene: dict[str, Any]
    gt_positions: np.ndarray
    predict_inputs: PredictorInputs
    fps: float


def _resolve_cameras(
    scene: dict[str, Any],
    camera: int,
    cameras: list[int] | str | None,
) -> list[int]:
    """Resolve and validate camera indices for prediction input creation."""
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


def _build_single_view_input(scene: dict[str, Any], camera_idx: int) -> PredictorInputs:
    """Build predictor input tensors from a single camera."""
    cam = scene["cameras"][camera_idx]
    return PredictorInputs(
        ball_uv=torch.from_numpy(cam["ball_uv"]).float(),
        court_kp=torch.from_numpy(cam["court_kp_uv"]).float(),
        ball_vis=torch.from_numpy(cam["ball_visible"].astype(np.float32)),
        court_vis=torch.from_numpy(cam["court_kp_visible"].astype(np.float32)),
        ball_mask=None,
    )


def _build_multiview_input(scene: dict[str, Any], cameras: list[int]) -> PredictorInputs:
    """Build predictor input tensors by stacking multiple cameras."""
    ball_uv_list: list[np.ndarray] = []
    court_kp_list: list[np.ndarray] = []
    ball_vis_list: list[np.ndarray] = []
    court_vis_list: list[np.ndarray] = []

    for camera_idx in cameras:
        cam = scene["cameras"][camera_idx]
        ball_uv_list.append(cam["ball_uv"])
        court_kp_list.append(cam["court_kp_uv"])
        ball_vis_list.append(cam["ball_visible"].astype(np.float32))
        court_vis_list.append(cam["court_kp_visible"].astype(np.float32))

    ball_vis = torch.from_numpy(np.stack(ball_vis_list, axis=0)).float()
    return PredictorInputs(
        ball_uv=torch.from_numpy(np.stack(ball_uv_list, axis=0)).float(),
        court_kp=torch.from_numpy(np.stack(court_kp_list, axis=0)).float(),
        ball_vis=ball_vis,
        court_vis=torch.from_numpy(np.stack(court_vis_list, axis=0)).float(),
        ball_mask=torch.ones_like(ball_vis),
    )


def load_scene_bundle(
    scene_path: Path,
    camera: int,
    cameras: list[int] | str | None,
) -> SceneBundle:
    """Load scene and prepare GT/predict artifacts.

    Args:
        scene_path: Path to scene npz file.
        camera: Fallback single camera index.
        cameras: Optional explicit camera list. If provided and length > 1,
            multi-view predictor inputs are generated.

    Returns:
        SceneBundle containing scene object, GT positions, predictor inputs and fps.
    """
    scene = load_scene(scene_path)
    gt_positions = scene["ball_pos_world"]
    fps = float(scene["meta"].get("fps_out", 30.0))

    selected_cameras = _resolve_cameras(scene, camera=camera, cameras=cameras)
    if len(selected_cameras) == 1:
        predict_inputs = _build_single_view_input(scene, selected_cameras[0])
    else:
        predict_inputs = _build_multiview_input(scene, selected_cameras)

    return SceneBundle(
        scene=scene,
        gt_positions=gt_positions,
        predict_inputs=predict_inputs,
        fps=fps,
    )
