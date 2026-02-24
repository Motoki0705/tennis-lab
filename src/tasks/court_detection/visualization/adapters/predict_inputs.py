"""Adapters for building court predictor inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tasks.court_detection.visualization.types import SceneImage


@dataclass(frozen=True)
class CourtPredictInputs:
    """Model-ready predictor input payload."""

    image_rgb: np.ndarray


def build_court_predict_inputs(scene: SceneImage) -> CourtPredictInputs:
    """Normalize scene image array for predictor inference."""
    image_rgb = np.asarray(scene.image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {tuple(image_rgb.shape)}")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)
    return CourtPredictInputs(image_rgb=image_rgb)
