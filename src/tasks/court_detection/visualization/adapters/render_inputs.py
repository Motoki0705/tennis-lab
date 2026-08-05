"""Adapters that convert training batch tensors into renderer inputs.

Used by the training LightningModule to feed raw model outputs into the
shared ``rendering/`` layer without duplicating image-decoding logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.tasks.court_detection.visualization.io.frames import (
    CourtFrame,
    KpFramePrediction,
)
from src.tasks.court_detection.visualization.rendering.common import (
    denormalize_tensor_to_rgb,
)


def batch_to_court_frame(
    batch: dict[str, Any],
    *,
    sample_idx: int = 0,
) -> CourtFrame:
    """Extract one sample from a batch dict and return a :class:`CourtFrame`.

    Args:
        batch: Training batch containing ``"image"`` key with shape
            ``(B, 3, H, W)`` ImageNet-normalized float tensor.
        sample_idx: Index of the sample within the batch to use.

    Returns:
        :class:`CourtFrame` with ``rgb`` as ``(H, W, 3)`` uint8 RGB.
    """
    img_tensor = batch["image"][sample_idx]  # (3, H, W)
    rgb = denormalize_tensor_to_rgb(img_tensor)
    return CourtFrame(name=f"sample_{sample_idx:02d}", rgb=rgb)


def logits_to_seg_mask(logits: torch.Tensor) -> np.ndarray:
    """Convert seg logits to a class mask.

    Args:
        logits: ``(C, H, W)`` float tensor (raw model output, C classes).

    Returns:
        ``(H, W)`` int NumPy array of class indices.
    """
    mask: np.ndarray = logits.argmax(dim=0).cpu().numpy().astype(np.int32)
    return mask


def logits_to_kp_prediction(logits: torch.Tensor) -> KpFramePrediction:
    """Convert kp logits to a :class:`KpFramePrediction`.

    Args:
        logits: ``(K, H, W)`` float tensor (raw model output, K keypoints).

    Returns:
        :class:`KpFramePrediction` with:
        - ``mean_heatmap``: ``(H, W)`` float ``[0, 1]`` -- max over K channels
          of the sigmoid-activated heatmap.
        - ``keypoints_px``: ``(K, 2)`` float array of ``(x, y)`` peak
          coordinates in pixel space.
    """
    prob = torch.sigmoid(logits).cpu()  # (K, H, W)

    # mean heatmap: max over keypoint channels so all peaks are visible
    mean_heatmap = prob.max(dim=0).values.numpy().astype(np.float32)  # (H, W)

    # peak coordinates per keypoint channel
    k, h, w = prob.shape
    flat_indices = prob.reshape(k, -1).argmax(dim=1)  # (K,)
    ys = (flat_indices // w).numpy().astype(np.float32)
    xs = (flat_indices % w).numpy().astype(np.float32)
    keypoints_px = np.stack([xs, ys], axis=1)  # (K, 2) as (x, y)

    return KpFramePrediction(keypoints_px=keypoints_px, mean_heatmap=mean_heatmap)


def logits_to_line_prob(logits: torch.Tensor) -> np.ndarray:
    """Convert line logits to a probability map.

    Args:
        logits: ``(1, H, W)`` float tensor (raw model output).

    Returns:
        ``(H, W)`` float NumPy array in ``[0, 1]``.
    """
    probability: np.ndarray = (
        torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
    )
    return probability
