"""Physical-coordinate trajectory refinement API."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.inference.windowing import predict_normalized
from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.tasks.ball_refiner_3d.model_io.checkpoint import load_checkpoint
from src.tasks.ball_refiner_3d.model_io.contracts import (
    RefinerPrediction,
    validate_input,
)
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel
from src.tasks.base.inference.predictor import BasePredictor
from src.utils.device import resolve_device


@torch.no_grad()  # type: ignore[untyped-decorator]
def refine_coordinates(
    model: RefinerModel,
    coordinates: Tensor,
    missing: Tensor,
    *,
    seed: int,
    batch_size: int = 32,
) -> RefinerPrediction:
    """Physical metres (3D), with true=missing; return all frames.

    Input is (V,T,D), where V is merely independent sequences, not a model view
    feature. FPS must match checkpoint metadata; resampling is a caller concern.
    """
    validate_input(coordinates, missing, model.config.dimensions)
    scale_np, offset_np = normalization(model.config.dimensions)
    scale = coordinates.new_tensor(scale_np)
    offset = coordinates.new_tensor(offset_np)
    normalized = torch.where(missing[..., None], 0, coordinates / scale - offset)
    result = predict_normalized(
        model, normalized, missing, batch_size=batch_size, seed=seed
    )
    return RefinerPrediction(
        (result.coordinates + offset) * scale, result.event_probability
    )


class RefinerPredictor(BasePredictor[RefinerPrediction]):
    """One checkpoint, physical metre input, CPU trajectory/event output."""

    def __init__(
        self, model: RefinerModel, metadata: dict[str, Any], device: torch.device
    ) -> None:
        self.model = model.to(device).eval()
        self.metadata = metadata
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, *, device: str | torch.device = "cpu"
    ) -> RefinerPredictor:
        if not checkpoint.is_absolute():
            raise ValueError("Predictor checkpoint must be an explicit absolute path")
        resolved = resolve_device(device)
        model, metadata = load_checkpoint(checkpoint, resolved)
        return cls(model, metadata, resolved)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def predict(
        self,
        coordinates: Tensor,
        missing: Tensor,
        *,
        fps: float,
        seed: int,
        batch_size: int = 32,
    ) -> RefinerPrediction:
        if not math.isclose(fps, self.metadata["fps"], abs_tol=1e-6, rel_tol=0):
            raise ValueError(
                "Input FPS does not match checkpoint FPS; resample explicitly"
            )
        result = refine_coordinates(
            self.model,
            coordinates.to(self.device),
            missing.to(self.device),
            seed=seed,
            batch_size=batch_size,
        )
        return RefinerPrediction(
            result.coordinates.cpu(), result.event_probability.cpu()
        )
