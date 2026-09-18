"""Strict model-only loading of Court training checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from src.tasks.court_detection.data.bundle_state import deserialize_target_bundle
from src.tasks.court_detection.model_io.factory import (
    CourtDetectionBoundModelIO,
    build_court_inference_pair,
)
from src.utils.configuration import PathResolver


def load_court_pair(path: Path, *, resolver: PathResolver, strict: bool = True,
                    weights_only: bool = False) -> CourtDetectionBoundModelIO:
    """Load every model tensor exactly; training-only state is not instantiated."""
    if not strict:
        raise ValueError("Court inference requires strict checkpoint loading")
    checkpoint = torch.load(path, map_location="cpu", weights_only=weights_only)
    hyperparameters = checkpoint["hyper_parameters"]
    bundle = deserialize_target_bundle(hyperparameters["target_bundle_state"])
    pair = build_court_inference_pair(hyperparameters["config"], resolver=resolver, target_bundle=bundle)
    model_weights = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items() if key.startswith("model.")
    }
    if not model_weights:
        raise ValueError("Court checkpoint contains no model state")
    pair.model.load_state_dict(model_weights, strict=True)
    return pair
