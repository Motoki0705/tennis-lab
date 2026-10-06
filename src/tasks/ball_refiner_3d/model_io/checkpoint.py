"""Strict inference loading for event-head checkpoints."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_refiner_3d.configuration.core import parse_section
from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel, build_refiner

CHECKPOINT_SCHEMA = "ball_refiner_3d.events.v1"


def checkpoint_model_config(payload: dict[str, Any]) -> ModelConfig:
    if payload.get("schema") not in {CHECKPOINT_SCHEMA, "ball_refiner_3d.events.v2"}:
        raise ValueError(
            "Require a 3D event-head checkpoint; older coordinate/GMM checkpoints are incompatible"
        )
    sigma = payload.get("event_sigma_frames")
    if (
        not isinstance(sigma, (float, int))
        or isinstance(sigma, bool)
        or not math.isfinite(sigma)
        or sigma <= 0
    ):
        raise ValueError("Checkpoint requires positive finite event_sigma_frames")
    return parse_section(ModelConfig, payload["model_config"])


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[RefinerModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint_model_config(payload)
    model: RefinerModel = build_refiner(config)
    if payload["schema"] == CHECKPOINT_SCHEMA:
        state = payload["model"]
    else:
        state = {
            key.removeprefix("model."): value
            for key, value in payload["state_dict"].items()
            if key.startswith("model.")
        }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def checkpoint_metadata(
    model: RefinerModel, *, event_sigma_frames: float
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": asdict(model.config),
        "event_sigma_frames": event_sigma_frames,
        "input_contract": "3D coordinates + missing boolean only; true=missing",
        "output_contract": "one all-frame 3D trajectory + per-frame softmax(no-event,event) probability",
    }
