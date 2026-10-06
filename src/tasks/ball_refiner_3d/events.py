"""Soft binary event targets and cross-entropy, independent of input corruption."""
import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def gaussian_event_target(events: np.ndarray, sigma_frames: float) -> NDArray[np.float32]:
    """Peak 1 at shot/bounce; max of full-rally Gaussian tails for overlaps."""
    if events.ndim != 1 or not math.isfinite(sigma_frames) or sigma_frames <= 0:
        raise ValueError("Require a 1D event sequence and positive finite sigma")
    result: NDArray[np.float32] = np.zeros(len(events), np.float32)
    time: NDArray[np.float64] = np.arange(len(events), dtype=np.float64)
    for index in np.flatnonzero(events):
        result = np.maximum(result, np.exp(-0.5 * ((time - index) / sigma_frames) ** 2).astype(np.float32))
    return result


def event_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Soft CE with class targets [1-p, p]; no temporal softmax or label threshold."""
    labels = torch.stack((1 - target, target), dim=-1)
    return -(labels * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
