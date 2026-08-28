"""Prediction API for BLCS visualization."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io import blcs_trajectory_prediction_to_physical
from src.utils.configuration import PathResolver, PathRole

logger = logging.getLogger(__name__)


def predict_positions(
    checkpoint_path: str | Path,
    resolver: PathResolver,
    device: str,
    scene: Mapping[str, Any],
    cameras: list[int],
    court_keypoint_contract: CourtKeypointContract,
) -> np.ndarray:
    """Run BLCS prediction and return denormalized 3D positions.

    Args:
        checkpoint_path: Path to BLCS checkpoint.
        device: Inference device (``cpu``/``cuda``/``auto``-resolved value).
        scene: Canonical BLCS scene mapping.
        cameras: Explicit selected camera indices.

    Returns:
        Predicted positions as ``(T, 3)`` numpy array in meters.
    """
    predictor = BLCSPredictor.load_from_checkpoint(
        checkpoint_path=Path(checkpoint_path).relative_to(
            resolver.roots.root(PathRole.CHECKPOINT)
        ),
        resolver=resolver,
        device=device,
        court_keypoints=court_keypoint_contract,
    )
    logger.info(f"Model loaded successfully on {device}.")
    raw_cameras = scene.get("cameras")
    if not isinstance(raw_cameras, list) or not cameras:
        raise ValueError("BLCS visualization requires selected scene cameras.")
    reference_camera_id: str | None = None
    if court_keypoint_contract.camera_view_semantics:
        raw_reference = raw_cameras[cameras[0]]
        if not isinstance(raw_reference, Mapping):
            raise TypeError("BLCS visualization camera must be a mapping.")
        value = raw_reference.get("camera_id")
        if not isinstance(value, str):
            raise ValueError("camera_view_v2 visualization requires stable camera ID.")
        reference_camera_id = value
    prediction = predictor.predict_scene(
        scene,
        cameras,
        denormalize=True,
        reference_camera_id=reference_camera_id,
    )
    prediction = blcs_trajectory_prediction_to_physical(prediction)
    logger.info("  [Inference] Running prediction...")
    position: np.ndarray = prediction.position.squeeze(0).numpy()
    return position
