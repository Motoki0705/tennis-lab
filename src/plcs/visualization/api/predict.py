"""Prediction API for PLCS visualization."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.plcs.inference.predictor import PLCSPredictor
from src.plcs.models.plcs_model import PLCSModel
from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.models.plcs_query_sequence_model import PLCSQuerySequenceModel


def _build_multiview_inputs(scene: Any, cameras: list[int]) -> dict[str, torch.Tensor]:
    human_kp = np.stack([scene.cameras[c].human_kp_uv for c in cameras], axis=0)
    court_kp = np.stack([scene.cameras[c].court_kp_uv for c in cameras], axis=0)
    human_vis = np.stack(
        [scene.cameras[c].human_kp_visible.astype(np.float32) for c in cameras],
        axis=0,
    )
    court_vis = np.stack(
        [scene.cameras[c].court_kp_visible.astype(np.float32) for c in cameras],
        axis=0,
    )
    human_mask = np.ones((human_kp.shape[0], human_kp.shape[1]), dtype=np.float32)

    return {
        "human_kp": torch.from_numpy(human_kp).float().unsqueeze(0),
        "court_kp": torch.from_numpy(court_kp).float().unsqueeze(0),
        "human_vis": torch.from_numpy(human_vis).float().unsqueeze(0),
        "human_mask": torch.from_numpy(human_mask).float().unsqueeze(0),
        "court_vis": torch.from_numpy(court_vis).float().unsqueeze(0),
    }


def _build_sequence_inputs(scene: Any, camera_idx: int) -> dict[str, torch.Tensor]:
    cam = scene.cameras[camera_idx]
    num_frames = int(cam.human_kp_uv.shape[0])
    return {
        "human_kp": torch.from_numpy(cam.human_kp_uv).float().unsqueeze(0),
        "court_kp": torch.from_numpy(cam.court_kp_uv).float().unsqueeze(0),
        "human_vis": torch.from_numpy(cam.human_kp_visible.astype(np.float32))
        .float()
        .unsqueeze(0),
        "human_mask": torch.ones((1, num_frames), dtype=torch.float32),
        "court_vis": torch.from_numpy(cam.court_kp_visible.astype(np.float32))
        .float()
        .unsqueeze(0),
    }


def _predict_frame_model(
    predictor: PLCSPredictor,
    scene: Any,
    camera_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    cam = scene.cameras[camera_idx]
    num_frames = int(cam.human_kp_uv.shape[0])
    pred_pos_list: list[np.ndarray] = []
    pred_rot_list: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        outputs = predictor.predict(
            human_kp=torch.from_numpy(cam.human_kp_uv[frame_idx]).float().unsqueeze(0),
            court_kp=torch.from_numpy(cam.court_kp_uv[frame_idx]).float().unsqueeze(0),
            human_vis=torch.from_numpy(cam.human_kp_visible[frame_idx].astype(np.float32))
            .float()
            .unsqueeze(0),
            human_mask=torch.ones((1,), dtype=torch.float32),
            court_vis=torch.from_numpy(cam.court_kp_visible[frame_idx].astype(np.float32))
            .float()
            .unsqueeze(0),
            denormalize=False,
        )
        pred_pos_list.append(outputs["position"].squeeze(0).numpy())
        pred_rot_list.append(outputs["rotation"].squeeze(0).numpy())

    return np.stack(pred_pos_list, axis=0), np.stack(pred_rot_list, axis=0)


def _predict_sequence_model(
    predictor: PLCSPredictor,
    scene: Any,
    camera_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = predictor.predict(
        denormalize=False,
        **_build_sequence_inputs(scene, camera_idx),
    )
    return outputs["position"].squeeze(0).numpy(), outputs["rotation"].squeeze(0).numpy()


def _predict_multiview_model(
    predictor: PLCSPredictor,
    scene: Any,
    cameras: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    outputs = predictor.predict(
        denormalize=False,
        **_build_multiview_inputs(scene, cameras),
    )
    pos = outputs["position"]
    rot = outputs["rotation"]
    if pos.dim() == 2:
        pos = pos.unsqueeze(1)
    if rot.dim() == 2:
        rot = rot.unsqueeze(1)
    return pos.squeeze(0).numpy(), rot.squeeze(0).numpy()


def predict_scene(
    checkpoint_path: str | Path,
    device: str,
    scene: Any,
    cameras: list[int],
) -> Any:
    """Run PLCS prediction and return a scene whose pose is replaced by prediction.

    Args:
        checkpoint_path: Path to PLCS checkpoint.
        device: Inference device.
        scene: Loaded PLCS scene object.
        cameras: Camera indices selected for prediction.

    Returns:
        Deep-copied scene with ``position`` and ``rotation`` replaced by prediction.
    """
    if not cameras:
        raise ValueError("No cameras selected for prediction.")

    predictor = PLCSPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model = predictor.model
    primary_camera = cameras[0]

    if isinstance(model, PLCSMultiViewModel):
        pred_pos, pred_rot = _predict_multiview_model(predictor, scene, cameras)
    elif isinstance(model, PLCSQuerySequenceModel):
        pred_pos, pred_rot = _predict_sequence_model(predictor, scene, primary_camera)
    elif isinstance(model, PLCSModel):
        pred_pos, pred_rot = _predict_frame_model(predictor, scene, primary_camera)
    else:
        raise ValueError(
            "Unsupported model type for PLCS visualization predict mode: "
            f"{type(model).__name__}"
        )

    predicted_scene = copy.deepcopy(scene)
    predicted_scene.position[...] = pred_pos
    predicted_scene.rotation[...] = pred_rot
    return predicted_scene
