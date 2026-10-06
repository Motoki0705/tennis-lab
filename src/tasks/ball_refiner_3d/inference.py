"""Explicit checkpoint loading and one full offline trajectory per input."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.config import (
    ModelConfig,
    parse_section,
)
from src.tasks.ball_refiner_3d.data import normalization
from src.tasks.ball_refiner_3d.model import CoordinateRefiner, validate_input
from src.tasks.ball_refiner_3d.models.contracts import RefinerPrediction
from src.tasks.ball_refiner_3d.temporal import window_owners, window_starts

CHECKPOINT_SCHEMA = "ball_refiner_3d.events.v1"
RefinerModel: TypeAlias = CoordinateRefiner


def checkpoint_model_config(payload: dict[str, Any]) -> ModelConfig:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Require a 3D event-head checkpoint; older coordinate/GMM checkpoints are incompatible")
    sigma = payload.get("event_sigma_frames")
    if not isinstance(sigma, (float, int)) or isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Checkpoint requires positive finite event_sigma_frames")
    return parse_section(ModelConfig, payload["model_config"])


def load_checkpoint(path: Path, device: torch.device) -> tuple[RefinerModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint_model_config(payload)
    model: RefinerModel = CoordinateRefiner(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_normalized(model: RefinerModel, coordinates: Tensor, missing: Tensor, *, batch_size: int, seed: int) -> RefinerPrediction:
    """V,T,D -> V,T,D. Real windows, nearest-center ownership, no mode averaging."""
    validate_input(coordinates, missing, model.config.dimensions)
    if batch_size < 1 or seed < 0:
        raise ValueError("Require positive batch size and nonnegative seed")
    frames = coordinates.shape[1]
    length = min(frames, model.config.window_length)
    starts = window_starts(frames, length, max(1, length // 2))
    owners = window_owners(frames, starts, length)
    output = torch.empty_like(coordinates)
    events = torch.empty_like(coordinates[..., 0])
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
                    output[view, indices + start] = prediction.coordinates[local, indices]
                    events[view, indices + start] = prediction.event_probability[local, indices]
    finally:
        model.train(was_training)
    if not torch.isfinite(output).all() or not torch.isfinite(events).all():
        raise ValueError("Refiner produced nonfinite coordinates")
    return RefinerPrediction(output, events)


@torch.no_grad()  # type: ignore[untyped-decorator]
def refine_coordinates(model: RefinerModel, coordinates: Tensor, missing: Tensor, *, seed: int, batch_size: int = 32) -> RefinerPrediction:
    """Physical metres (3D), with true=missing; return all frames.

    Input is (V,T,D), where V is merely independent sequences, not a model view
    feature. FPS must match checkpoint metadata; resampling is a caller concern.
    """
    validate_input(coordinates, missing, model.config.dimensions)
    scale_np, offset_np = normalization(model.config.dimensions)
    scale = coordinates.new_tensor(scale_np)
    offset = coordinates.new_tensor(offset_np)
    normalized = torch.where(missing[..., None], 0, coordinates / scale - offset)
    result = predict_normalized(model, normalized, missing, batch_size=batch_size, seed=seed)
    return RefinerPrediction((result.coordinates + offset) * scale, result.event_probability)


def checkpoint_metadata(model: RefinerModel, *, event_sigma_frames: float) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "model_config": asdict(model.config),
            "event_sigma_frames": event_sigma_frames,
            "input_contract": "3D coordinates + missing boolean only; true=missing",
            "output_contract": "one all-frame 3D trajectory + per-frame softmax(no-event,event) probability"}
