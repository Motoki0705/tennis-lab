"""Prediction API for PLCS visualization."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.utils.configuration import PathResolver
from src.utils.schema.court_normalization import CourtCoordinateNormalization

logger = logging.getLogger(__name__)

CanonicalPoseSource = Literal["gt", "prediction"]


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
    court_coordinate_normalization: CourtCoordinateNormalization,
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
        court_coordinate_normalization=court_coordinate_normalization,
    )
    logger.info(f"Model loaded successfully on {device}.")
    decoded = predictor.predict_scene(scene, cameras)
    pred_pos = decoded.position.squeeze(0).numpy()
    pred_rot = decoded.rotation.squeeze(0).numpy()
    pred_canonical_pose = (
        decoded.canonical_pose.squeeze(0).numpy()
        if decoded.canonical_pose is not None
        else None
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
