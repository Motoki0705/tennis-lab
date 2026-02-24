"""Prediction API helpers for court detection visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tasks.court_detection.visualization.adapters.predict_inputs import CourtPredictInputs
from src.tasks.court_detection.visualization.types import KeypointPrediction


def load_predictor(*, checkpoint_path: Path, device: str) -> CourtKeypointPredictor:
    """Load court keypoint predictor from checkpoint."""
    return CourtKeypointPredictor.load_from_checkpoint(checkpoint_path, device=device)


def predict_keypoints(
    *,
    predictor: CourtKeypointPredictor,
    inputs: CourtPredictInputs,
) -> KeypointPrediction:
    """Run keypoint prediction and return numpy outputs."""
    outputs = predictor.predict(inputs.image_rgb)
    keypoints = outputs["keypoints"].detach().cpu().numpy().astype(np.float32)
    visibility = outputs["visibility"].detach().cpu().numpy().astype(np.float32)
    return KeypointPrediction(keypoints=keypoints, visibility=visibility)
