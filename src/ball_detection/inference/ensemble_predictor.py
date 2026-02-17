"""Ensemble predictor combining local and WASB-origin models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.ball_detection.inference.predictor import BallPredictor


class BallEnsemblePredictor:
    """Weighted average ensemble over multiple predictors with aligned output contract."""

    def __init__(self, predictors: list[BallPredictor], weights: list[float] | None = None) -> None:
        if not predictors:
            raise ValueError("predictors must be non-empty")
        self.predictors = predictors
        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            if len(weights) != len(predictors):
                raise ValueError("weights length must match predictors length")
            s = sum(weights)
            if s <= 0:
                raise ValueError("weights sum must be positive")
            self.weights = [float(w) / s for w in weights]

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str | Path],
        *,
        device: str = "cpu",
        weights: list[float] | None = None,
    ) -> "BallEnsemblePredictor":
        predictors = [BallPredictor.load_from_checkpoint(path, device=device) for path in checkpoint_paths]
        return cls(predictors=predictors, weights=weights)

    @torch.no_grad()
    def predict(self, frames: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        _ = kwargs
        per_model = [p.predict(frames) for p in self.predictors]

        xy = torch.zeros_like(per_model[0]["ball_uv"])
        score = torch.zeros_like(per_model[0]["score"])
        for w, pred in zip(self.weights, per_model, strict=True):
            xy = xy + pred["ball_uv"] * float(w)
            score = score + pred["score"] * float(w)

        visibility = (score >= 0.5).to(torch.float32)
        return {
            "ball_uv": xy,
            "visibility": visibility,
            "score": score,
        }
