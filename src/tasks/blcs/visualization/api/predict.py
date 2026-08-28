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
    reference_camera_id: str | None,
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
    if court_keypoint_contract.camera_view_semantics:
        if not isinstance(reference_camera_id, str) or not reference_camera_id.strip():
            raise ValueError(
                "camera_view_v2 visualization requires an explicit stable "
                "reference_camera_id."
            )
    elif reference_camera_id is not None:
        raise ValueError("physical_v1 visualization must not specify a reference camera.")
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
