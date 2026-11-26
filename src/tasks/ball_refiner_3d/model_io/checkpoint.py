"""Strict inference loading for event-head checkpoints."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_refiner_3d.configuration.core import parse_section
from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel, build_refiner
from src.tasks.ball_refiner_3d.physics.targets import FlightClock

CHECKPOINT_SCHEMA = "ball_refiner_3d.events.v3"
LEGACY_EVENT_SCHEMAS = {"ball_refiner_3d.events.v1", "ball_refiner_3d.events.v2"}


def checkpoint_model_config(payload: dict[str, Any]) -> ModelConfig:
    schema = payload.get("schema")
    if schema in LEGACY_EVENT_SCHEMAS:
        raise ValueError(
            f"{schema} has one event channel; require a three-class {CHECKPOINT_SCHEMA} checkpoint"
        )
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Require a 3D event-head checkpoint with schema {CHECKPOINT_SCHEMA}"
        )
    if payload.get("event_classes") != list(EVENT_CLASSES):
        raise ValueError(
            f"Checkpoint event_classes must be {list(EVENT_CLASSES)} in order"
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
    if ("model" in payload) == ("state_dict" in payload):
        raise ValueError(
            "Checkpoint must contain either model or Lightning state_dict weights"
        )
    if "model" in payload:
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


def checkpoint_flight_clock(payload: dict[str, Any]) -> FlightClock:
    """Integration constants of the training data, required by physics heads."""
    if "flight_clock" not in payload:
        raise ValueError("Checkpoint has no flight clock for physics integration")
    clock: FlightClock = parse_section(FlightClock, payload["flight_clock"])
    return clock


def checkpoint_metadata(
    model: RefinerModel, *, event_sigma_frames: float, clock: FlightClock
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "flight_clock": asdict(clock),
        "model_config": asdict(model.config),
        "event_classes": list(EVENT_CLASSES),
        "event_sigma_frames": event_sigma_frames,
        "input_contract": "3D coordinates + missing boolean only; true=missing",
        "output_contract": "one all-frame 3D trajectory + independent per-frame (shot,bounce,net) probabilities"
        + (
            " + rally field/surface and flight-segment initial states"
            if model.config.physics_heads
            else ""
        ),
    }
