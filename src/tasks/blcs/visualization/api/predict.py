"""Prediction API for BLCS visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.tasks.blcs.visualization.adapters.predict_inputs import PredictorInputs
from src.tasks.blcs.inference.predictor import BLCSPredictor

logger = logging.getLogger(__name__)


def predict_positions(
    checkpoint_path: str | Path,
    device: str,
    inputs: PredictorInputs,
) -> np.ndarray:
    """Run BLCS prediction and return denormalized 3D positions.

    Args:
        checkpoint_path: Path to BLCS checkpoint.
        device: Inference device (``cpu``/``cuda``/``auto``-resolved value).
        inputs: Predictor tensors prepared from scene cameras.

    Returns:
        Predicted positions as ``(T, 3)`` numpy array in meters.
    """
    predictor = BLCSPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    logger.info(f"Model loaded successfully on {device}.")
    outputs = predictor.predict(
        ball_uv=inputs.ball_uv,
        court_kp=inputs.court_kp,
        ball_vis=inputs.ball_vis,
        ball_mask=inputs.ball_mask,
        court_vis=inputs.court_vis,
        denormalize=True,
    )
    logger.info("  [Inference] Running prediction...")
    return outputs["position"].squeeze(0).cpu().numpy()
