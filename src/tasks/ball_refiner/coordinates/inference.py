"""Explicit checkpoint loading and one full offline trajectory per input."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_refiner.coordinates.config import (
    LegacyModelConfig,
    ModelConfig,
    parse_section,
)
from src.tasks.ball_refiner.coordinates.data import normalization
from src.tasks.ball_refiner.coordinates.model import CoordinateRefiner, validate_input
from src.tasks.ball_refiner.coordinates.models.legacy import LegacyCoordinateRefiner
from src.tasks.ball_refiner.data.temporal import window_owners, window_starts

CHECKPOINT_SCHEMA = "ball_refiner.coordinates.v2"
LEGACY_CHECKPOINT_SCHEMA = "ball_refiner.coordinates.v1"
RefinerModel = CoordinateRefiner | LegacyCoordinateRefiner


def checkpoint_model_config(payload: dict[str, Any]) -> ModelConfig | LegacyModelConfig:
    if payload["schema"] == CHECKPOINT_SCHEMA:
        return parse_section(ModelConfig, payload["model_config"])
    if payload["schema"] == LEGACY_CHECKPOINT_SCHEMA:
        return parse_section(LegacyModelConfig, payload["model_config"])
    raise ValueError("Expected a coordinate refiner checkpoint; a legacy GMM bundle is incompatible")


def load_checkpoint(path: Path, device: torch.device) -> tuple[RefinerModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint_model_config(payload)
    model: RefinerModel = CoordinateRefiner(config) if isinstance(config, ModelConfig) else LegacyCoordinateRefiner(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_normalized(model: RefinerModel, coordinates: Tensor, missing: Tensor, *, batch_size: int, seed: int) -> Tensor:
    """V,T,D -> V,T,D. Real windows, nearest-center ownership, no mode averaging."""
    validate_input(coordinates, missing, model.config.dimensions)
    if batch_size < 1 or seed < 0:
        raise ValueError("Require positive batch size and nonnegative seed")
    frames = coordinates.shape[1]
    length = min(frames, model.config.window_length)
    starts = window_starts(frames, length, max(1, length // 2))
    owners = window_owners(frames, starts, length)
    output = torch.empty_like(coordinates)
    generator = torch.Generator(device=coordinates.device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for view in range(len(coordinates)):
            for first in range(0, len(starts), batch_size):
                selected = starts[first:first + batch_size]
                inputs = torch.stack([coordinates[view, start:start + length] for start in selected])
                mask = torch.stack([missing[view, start:start + length] for start in selected])
                prediction = model.predict(inputs, mask, generator=generator)
                for local, start in enumerate(selected):
                    take = np.flatnonzero(owners[start:start + length] == first + local)
                    indices = torch.from_numpy(take).to(coordinates.device)
                    output[view, indices + start] = prediction[local, indices]
    finally:
        model.train(was_training)
    if not torch.isfinite(output).all():
        raise ValueError("Refiner produced nonfinite coordinates")
    return output


@torch.no_grad()  # type: ignore[untyped-decorator]
def refine_coordinates(model: RefinerModel, coordinates: Tensor, missing: Tensor, *, seed: int, batch_size: int = 32) -> Tensor:
    """Physical pixels (2D) or metres (3D), with true=missing; return all frames.

    Input is (V,T,D), where V is merely independent sequences, not a model view
    feature. FPS must match checkpoint metadata; resampling is a caller concern.
    """
    validate_input(coordinates, missing, model.config.dimensions)
    scale_np, offset_np = normalization(model.config.dimensions)
    scale = coordinates.new_tensor(scale_np)
    offset = coordinates.new_tensor(offset_np)
    normalized = torch.where(missing[..., None], 0, coordinates / scale - offset)
    return (predict_normalized(model, normalized, missing, batch_size=batch_size, seed=seed) + offset) * scale


def checkpoint_metadata(model: RefinerModel) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA if isinstance(model.config, ModelConfig) else LEGACY_CHECKPOINT_SCHEMA, "model_config": asdict(model.config),
            "input_contract": "coordinates + missing boolean only; true=missing", "output_contract": "one trajectory, all frames; absolute coordinates"}
