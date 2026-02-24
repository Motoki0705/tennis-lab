"""Prediction API for PLCS visualization."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.plcs.visualization.adapters.predict_inputs import (
    build_frame_inputs,
    build_multiview_inputs,
    build_sequence_inputs,
)
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.models.plcs_model import PLCSModel
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.tasks.plcs.models.plcs_query_sequence_model import PLCSQuerySequenceModel

logger = logging.getLogger(__name__)


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
            **build_frame_inputs(scene, camera_idx, frame_idx),
            denormalize=False,
        )
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0 or frame_idx == num_frames - 1:
            logger.info(f"  [Inference] Processing frame {frame_idx + 1}/{num_frames}...")
        pred_pos_list.append(outputs["position"].squeeze(0).numpy())
        pred_rot_list.append(outputs["rotation"].squeeze(0).numpy())

    return np.stack(pred_pos_list, axis=0), np.stack(pred_rot_list, axis=0)


def _predict_sequence_model(
    predictor: PLCSPredictor,
    scene: Any,
    camera_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("  [Inference] Running sequence model inference...")
    outputs = predictor.predict(
        denormalize=False,
        **build_sequence_inputs(scene, camera_idx),
    )
    return outputs["position"].squeeze(0).numpy(), outputs["rotation"].squeeze(0).numpy()


def _predict_multiview_model(
    predictor: PLCSPredictor,
    scene: Any,
    cameras: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("  [Inference] Running multiview model inference...")
    outputs = predictor.predict(
        denormalize=False,
        **build_multiview_inputs(scene, cameras),
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
    logger.info(f"Model loaded successfully on {device}.")
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
