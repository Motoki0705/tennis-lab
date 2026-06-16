"""Prediction API for court-detection visualization.

One entry point per task (``kp`` / ``seg`` / ``line``); each loads the matching
predictor once and runs it over every frame, returning per-frame results in a
display-agnostic form (original-pixel keypoints, averaged heatmap, or a dense
mask at model resolution) for the renderers to draw.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.tasks.court_detection.inference import (
    CourtKeypointPredictor,
    CourtLinePredictor,
    CourtSegPredictor,
)
from src.tasks.court_detection.visualization.adapters.predict_inputs import (
    to_predictor_input,
)
from src.tasks.court_detection.visualization.io.frames import (
    CourtFrame,
    KpFramePrediction,
)

logger = logging.getLogger(__name__)


def predict_kp(
    *,
    checkpoint_path: str | Path,
    device: str,
    frames: list[CourtFrame],
) -> list[KpFramePrediction]:
    """Predict keypoints + averaged heatmap for every frame."""
    predictor = CourtKeypointPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    logger.info("Court keypoint model loaded on %s.", device)
    predictions: list[KpFramePrediction] = []
    for index, frame in enumerate(frames):
        outputs = predictor.predict(to_predictor_input(frame), return_heatmaps=True)
        mean_heatmap = torch.sigmoid(outputs["heatmaps"]).mean(0).numpy()
        predictions.append(
            KpFramePrediction(
                keypoints_px=outputs["keypoints"].numpy(),
                mean_heatmap=mean_heatmap,
            )
        )
        _log_progress(index, len(frames))
    return predictions


def predict_seg(
    *,
    checkpoint_path: str | Path,
    device: str,
    frames: list[CourtFrame],
) -> list[np.ndarray]:
    """Predict a class mask ``(h', w')`` (int) for every frame."""
    predictor = CourtSegPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    logger.info("Court segmentation model loaded on %s.", device)
    masks: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        outputs = predictor.predict(to_predictor_input(frame))
        masks.append(outputs["seg_mask"].numpy().astype(np.uint8))
        _log_progress(index, len(frames))
    return masks


def predict_line(
    *,
    checkpoint_path: str | Path,
    device: str,
    frames: list[CourtFrame],
) -> list[np.ndarray]:
    """Predict a line-probability map ``(h', w')`` (float) for every frame."""
    predictor = CourtLinePredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    logger.info("Court line model loaded on %s.", device)
    probs: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        outputs = predictor.predict(to_predictor_input(frame))
        probs.append(outputs["line_prob"].numpy())
        _log_progress(index, len(frames))
    return probs


def _log_progress(index: int, total: int) -> None:
    if (index + 1) % 10 == 0 or index == 0 or index == total - 1:
        logger.info("  [Inference] Processing frame %d/%d...", index + 1, total)
