"""Prediction API for ball-detection visualization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.visualization.adapters.predict_inputs import (
    build_window_starts,
    iter_window_batches,
)
from src.tasks.ball_detection.visualization.io.clip import ClipSequence
from src.utils.data.heatmaps import heatmaps_to_argmax

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionSequence:
    """Aggregated per-frame predictions for one clip."""

    heatmaps: torch.Tensor
    coords_px: torch.Tensor
    confidences: torch.Tensor
    visibility: torch.Tensor


def predict_clip(
    *,
    predictor: BallDetectionPredictor,
    clip: ClipSequence,
    sequence_length: int,
    window_stride: int,
    inference_batch_size: int,
    image_size_hw: tuple[int, int],
    peak_threshold: float,
) -> PredictionSequence:
    """Run overlapping-window inference and aggregate per-frame predictions."""
    window_starts = build_window_starts(
        frame_count=len(clip.frame_names),
        sequence_length=sequence_length,
        stride=window_stride,
    )
    logger.info("Running predictor over %d overlapping window(s).", len(window_starts))

    heatmap_sum: torch.Tensor | None = None
    heatmap_count = torch.zeros(len(clip.frame_names), dtype=torch.float32)

    for start_chunk, batch in iter_window_batches(
        model_images=clip.model_images,
        window_starts=window_starts,
        sequence_length=sequence_length,
        batch_size=inference_batch_size,
    ):
        prediction = predictor.predict(batch)
        batch_heatmaps = prediction.heatmaps.to(torch.float32)

        if heatmap_sum is None:
            heatmap_sum = torch.zeros(
                (len(clip.frame_names), *batch_heatmaps.shape[-2:]),
                dtype=torch.float32,
            )

        for window_index, start in enumerate(start_chunk):
            end = start + sequence_length
            heatmap_sum[start:end] += batch_heatmaps[window_index]
            heatmap_count[start:end] += 1.0

    if heatmap_sum is None:
        raise RuntimeError("Failed to aggregate prediction heatmaps for the clip.")

    averaged_heatmaps = heatmap_sum / torch.clamp(heatmap_count, min=1.0).view(-1, 1, 1)
    coords_normalized, confidences = heatmaps_to_argmax(averaged_heatmaps)

    image_height, image_width = image_size_hw
    coords_px = torch.empty_like(coords_normalized)
    coords_px[:, 0] = coords_normalized[:, 0] * max(image_width - 1, 0)
    coords_px[:, 1] = coords_normalized[:, 1] * max(image_height - 1, 0)
    visibility = confidences >= peak_threshold

    return PredictionSequence(
        heatmaps=averaged_heatmaps,
        coords_px=coords_px,
        confidences=confidences,
        visibility=visibility,
    )


def build_mdd_frames(
    *,
    predictor: BallDetectionPredictor,
    clip: ClipSequence,
) -> list[np.ndarray]:
    """Build per-frame MDD RGB visualizations for the whole clip.

    The MDD features are computed over the full consecutive frame sequence so
    the panel reflects what the model derives as motion (green=brighten,
    red=darken). Computed from the same preprocessed images the predictor
    consumes to stay faithful to the model input.
    """
    with torch.no_grad():
        features = predictor.adapter.mdd_features(clip.model_images.unsqueeze(0))
    brighten = features[0, 0].clamp(0.0, 1.0).cpu().numpy()
    darken = features[0, 1].clamp(0.0, 1.0).cpu().numpy()

    mdd_frames: list[np.ndarray] = []
    for frame_index in range(brighten.shape[0]):
        rgb = np.zeros((*brighten.shape[1:], 3), dtype=np.uint8)
        rgb[..., 0] = (darken[frame_index] * 255.0).astype(np.uint8)
        rgb[..., 1] = (brighten[frame_index] * 255.0).astype(np.uint8)
        mdd_frames.append(rgb)
    return mdd_frames
