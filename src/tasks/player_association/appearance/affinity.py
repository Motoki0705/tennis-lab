"""Appearance evidence that two track segments of different cameras are one person.

A segment's appearance is the normalized mean of its sampled crop embeddings.
The score is a calibrated log-likelihood ratio of the cosine similarity,
``slope * (cosine - center)``, clipped to ``max_abs_score``. The calibration
comes from cross-camera pairs only: the same person seen from one camera looks
more alike than from two, so the score is not applied to same-camera pairs.

``fit_cosine_log_likelihood_ratio`` fits the two parameters by a class-balanced
logistic regression: with equal class weights the fitted logit estimates the
log-likelihood ratio rather than the posterior log-odds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class AppearanceAffinityConfig:
    encoder: str
    slope: float
    center: float
    max_abs_score: float

    def __post_init__(self) -> None:
        if not (self.encoder and self.slope > 0 and -1 <= self.center <= 1 and self.max_abs_score > 0):
            raise ValueError(f"Invalid appearance affinity config: {self}")


def segment_embedding(frames: NDArray[np.int64], embeddings: NDArray[np.float32], start: int, end: int) -> NDArray[np.float64] | None:
    """Unit mean embedding of the samples in ``[start, end)``, or ``None`` without samples."""
    if embeddings.ndim != 2 or frames.shape != (len(embeddings),):
        raise ValueError("Sample frames (K,) must align to embeddings (K, E)")
    inside = (frames >= start) & (frames < end)
    if not inside.any():
        return None
    mean = embeddings[inside].astype(np.float64).mean(0)
    norm = float(np.linalg.norm(mean))
    if not norm > 0:
        raise ValueError("Sample embeddings average to zero")
    result: NDArray[np.float64] = mean / norm
    return result


def appearance_score(cosine: float, config: AppearanceAffinityConfig) -> float:
    if not -1 - 1e-6 <= cosine <= 1 + 1e-6:
        raise ValueError(f"Cosine similarity out of range: {cosine}")
    return float(np.clip(config.slope * (cosine - config.center), -config.max_abs_score, config.max_abs_score))


def fit_cosine_log_likelihood_ratio(positive: NDArray[np.float64], negative: NDArray[np.float64]) -> tuple[float, float]:
    """``(slope, center)`` of a class-balanced logistic regression of same-person on cosine."""
    positive, negative = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    if positive.ndim != 1 or negative.ndim != 1 or len(positive) < 2 or len(negative) < 2:
        raise ValueError("The appearance fit needs at least two positive and two negative cosines")
    values = np.concatenate((positive, negative))
    targets = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    weights = np.concatenate((np.full(len(positive), .5 / len(positive)), np.full(len(negative), .5 / len(negative))))

    def loss(parameters: NDArray[np.float64]) -> float:
        logits = parameters[0] * values + parameters[1]
        return float((weights * (np.logaddexp(0, logits) - targets * logits)).sum())

    result = minimize(loss, np.zeros(2), method="BFGS")
    slope, intercept = float(result.x[0]), float(result.x[1])
    if not result.success or not slope > 0:
        raise RuntimeError(f"Appearance calibration failed (success={result.success}, slope={slope}): {result.message}")
    return slope, -intercept / slope
