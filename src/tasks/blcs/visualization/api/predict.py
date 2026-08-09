"""Prediction API for BLCS visualization."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.utils.configuration import PathResolver, PathRole

logger = logging.getLogger(__name__)


def predict_positions(
    checkpoint_path: str | Path,
    resolver: PathResolver,
    device: str,
    scene: Mapping[str, Any],
    cameras: list[int],
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
    )
    logger.info(f"Model loaded successfully on {device}.")
    prediction = predictor.predict_scene(
        scene,
        cameras,
        denormalize=True,
    )
    logger.info("  [Inference] Running prediction...")
    position: np.ndarray = prediction.position.squeeze(0).numpy()
    return position
