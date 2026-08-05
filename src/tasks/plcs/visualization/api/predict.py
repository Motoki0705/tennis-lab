"""Prediction API for PLCS visualization."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.models.plcs_model import PLCSModel
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.tasks.plcs.visualization.adapters.predict_inputs import (
    build_frame_inputs,
    build_multiview_inputs,
)
from src.utils.configuration import PathResolver

logger = logging.getLogger(__name__)

CanonicalPoseSource = Literal["gt", "prediction"]


def _predict_frame_model(
    predictor: PLCSPredictor,
    scene: Any,
    camera_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cam = scene.cameras[camera_idx]
    num_frames = int(cam.human_kp_uv.shape[0])
    pred_pos_list: list[np.ndarray] = []
    pred_rot_list: list[np.ndarray] = []
    pred_canonical_pose_list: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        outputs = predictor.predict(
            **build_frame_inputs(scene, camera_idx, frame_idx),
            denormalize=False,
        )
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0 or frame_idx == num_frames - 1:
            logger.info(
                f"  [Inference] Processing frame {frame_idx + 1}/{num_frames}..."
            )
        pred_pos_list.append(outputs["position"].squeeze(0).numpy())
        pred_rot_list.append(outputs["rotation"].squeeze(0).numpy())
        canonical_pose = outputs.get("canonical_pose")
        if canonical_pose is not None:
            pred_canonical_pose_list.append(canonical_pose.squeeze(0).numpy())

    pred_canonical_pose = None
    if pred_canonical_pose_list:
        pred_canonical_pose = np.stack(pred_canonical_pose_list, axis=0)
    return (
        np.stack(pred_pos_list, axis=0),
        np.stack(pred_rot_list, axis=0),
        pred_canonical_pose,
    )


def _predict_multiview_model(
    predictor: PLCSPredictor,
    scene: Any,
    cameras: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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

    canonical_pose = outputs.get("canonical_pose")
    if canonical_pose is not None and canonical_pose.dim() == 3:
        canonical_pose = canonical_pose.unsqueeze(1)

    return (
        pos.squeeze(0).numpy(),
        rot.squeeze(0).numpy(),
        canonical_pose.squeeze(0).numpy() if canonical_pose is not None else None,
    )


def _apply_canonical_pose_source(
    scene: Any,
    predicted_canonical_pose: np.ndarray | None,
    source: CanonicalPoseSource,
) -> None:
    """Keep GT canonical pose or replace it with the model prediction."""
    if source == "gt":
        return
    if source != "prediction":
        raise ValueError(
            f"canonical_pose_source must be 'gt' or 'prediction', got '{source}'."
        )
    if predicted_canonical_pose is None:
        raise ValueError(
            "canonical_pose_source='prediction' requires a model that outputs "
            "canonical_pose."
        )

    predicted_shape = predicted_canonical_pose.shape
    if len(predicted_shape) != 3 or predicted_shape[-1] != 3:
        raise ValueError(
            "Predicted canonical_pose must have shape (T, J, 3), "
            f"got {predicted_shape}."
        )
    existing = getattr(scene, "canonical_pose_3d", None)
    if existing is not None:
        existing_shape = np.asarray(existing).shape
        if not existing_shape or existing_shape[0] != predicted_shape[0]:
            raise ValueError(
                "Predicted canonical_pose frame count must match the scene: "
                f"expected {existing_shape[0] if existing_shape else 0}, "
                f"got {predicted_shape[0]}."
            )
    scene.canonical_pose_3d = predicted_canonical_pose.copy()


def predict_scene(
    checkpoint_path: str | Path,
    device: str,
    scene: Any,
    cameras: list[int],
    canonical_pose_source: CanonicalPoseSource = "gt",
    *,
    resolver: PathResolver,
) -> Any:
    """Run PLCS prediction and return a scene whose pose is replaced by prediction.

    Args:
        checkpoint_path: Path to PLCS checkpoint.
        device: Inference device.
        scene: Loaded PLCS scene object.
        cameras: Camera indices selected for prediction.
        canonical_pose_source: Canonical pose used to render the predicted
            position/rotation. ``"gt"`` preserves the input scene pose;
            ``"prediction"`` requires a canonical pose model output.

    Returns:
        Deep-copied scene with ``position`` and ``rotation`` replaced by prediction.
    """
    if not cameras:
        raise ValueError("No cameras selected for prediction.")

    predictor = PLCSPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        resolver=resolver,
        device=device,
        allow_device_fallback=False,
    )
    logger.info(f"Model loaded successfully on {device}.")
    model = predictor.model
    primary_camera = cameras[0]

    if isinstance(model, (PLCSMultiViewModel, PLCSMultiViewAxialModel)):
        pred_pos, pred_rot, pred_canonical_pose = _predict_multiview_model(
            predictor,
            scene,
            cameras,
        )
    elif isinstance(model, PLCSModel):
        pred_pos, pred_rot, pred_canonical_pose = _predict_frame_model(
            predictor,
            scene,
            primary_camera,
        )
    else:
        raise ValueError(
            "Unsupported model type for PLCS visualization predict mode: "
            f"{type(model).__name__}"
        )

    predicted_scene = copy.deepcopy(scene)
    predicted_scene.position[...] = pred_pos
    predicted_scene.rotation[...] = pred_rot
    _apply_canonical_pose_source(
        predicted_scene,
        pred_canonical_pose,
        canonical_pose_source,
    )
    return predicted_scene
