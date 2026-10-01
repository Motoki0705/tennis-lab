"""Deterministic, full-mixture uncertainty for consumers requiring one point.

The point is the maximum-weight component mean (first component on ties).
M = E[(X-point)(X-point)^T] includes within- and between-component uncertainty.
Markov's inequality gives P((X-point)^T M^-1 (X-point) < 20) >= .9 in 2D.
The ellipse area is 20*pi*sqrt(det(M)), conditional on presence, on all of R².
This conservative region is NOT the mixture HDR and is not clipped to the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D


@dataclass(frozen=True)
class PointConfidenceRule:
    min_presence: float
    max_area_px2: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_presence) or not 0 <= self.min_presence <= 1:
            raise ValueError("Confidence presence must be finite in [0,1]")
        if not math.isfinite(self.max_area_px2) or self.max_area_px2 <= 0:
            raise ValueError("Confidence area must be finite and positive in source px²")

    def rejection_codes(
        self, presence: NDArray[np.float64], area_px2: NDArray[np.float64],
    ) -> NDArray[np.uint8]:
        if presence.shape != area_px2.shape or not np.isfinite(presence).all() or not np.isfinite(area_px2).all():
            raise ValueError("Confidence arrays must have equal shapes and finite values")
        if np.any((presence < 0) | (presence > 1)) or np.any(area_px2 <= 0):
            raise ValueError("Invalid confidence probabilities or areas")
        # Zero means accepted; bits 1/2 distinguish the two explicit missing reasons.
        return ((presence < self.min_presence).astype(np.uint8)
                | ((area_px2 > self.max_area_px2).astype(np.uint8) << 1))


def point_confidence(
    prediction: BallGMM2D, source_size_wh: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (point_px, presence, conservative_area_px2), preserving B,T axes."""
    if len(source_size_wh) != 2 or any(type(x) is not int or x < 2 for x in source_size_wh):
        raise ValueError("Confidence requires explicit source width/height >=2")
    # CPU float64 makes the per-frame rule independent of batching, seed and CUDA.
    scale = torch.tensor(source_size_wh, dtype=torch.float64) - 1
    means = prediction.means.detach().cpu().double() * scale
    tril = prediction.scale_tril.detach().cpu().double() * scale[:, None]
    logits = prediction.mixture_logits.detach().cpu().double()
    weights = logits.softmax(-1)
    index = logits.argmax(-1)
    point = means.gather(-2, index[..., None, None].expand(*index.shape, 1, 2)).squeeze(-2)
    delta = means - point[..., None, :]
    second_moment = ((tril @ tril.transpose(-2, -1) + delta[..., :, None] * delta[..., None, :])
                     * weights[..., None, None]).sum(-3)
    sign, logdet = torch.linalg.slogdet(second_moment)
    if not bool((sign > 0).all()):
        raise ValueError("Mixture second moment must be positive definite")
    area = (20 * math.pi) * (logdet / 2).exp()
    presence = prediction.presence_logits.detach().cpu().double().sigmoid()
    if not bool(torch.isfinite(point).all() and torch.isfinite(area).all()):
        raise ValueError("Nonfinite confidence point or area")
    return point.numpy(), presence.numpy(), area.numpy()
