"""Inference-only checkpoint loading for the ball-detection review UI.

``BallDetectionPredictor.load_from_checkpoint`` restores a checkpoint through
``LightningModule.load_from_checkpoint``, which validates the *complete*
``BallDetectionTrainingConfig``.  Curated inference checkpoints predate parts of
that training boundary (for example ``data.split.root_role``), so this loader
instead rebuilds only the slice the model factory consumes:

1. ``evaluation.configuration.read_checkpoint_config`` reads the saved Hydra
   config with a memory-mapped ``torch.load`` without constructing the model.
2. ``build_ball_detection_pair`` picks and validates the architecture/adapter
   pair from ``model.name``.
3. Only the explicit ``model.`` state-dict entries are restored with
   ``strict=True``, so a missing or surplus parameter fails loudly instead of
   being patched or re-keyed.

Optimizer, scheduler, loss, and metric state are never read.  Nothing is
defaulted on the caller's behalf: a config without a resolvable input size
raises instead of guessing one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch import nn

from src.tasks.ball_detection.evaluation.configuration import read_checkpoint_config
from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from src.tasks.ball_detection.visualization.review.checkpoints import (
    MINIMUM_FRAMES_BY_MODEL,
)
from src.utils.device import resolve_device

#: Lightning stores module state under this prefix; only these entries are
#: model weights.
MODEL_STATE_PREFIX = "model."


class BallInferenceCheckpointError(ValueError):
    """Raised when a curated checkpoint cannot drive ball inference."""


@dataclass(frozen=True, slots=True)
class LoadedBallModel:
    """One constructed, weight-restored model bound to its I/O adapter."""

    model: nn.Module
    adapter: BallModelIOAdapter
    device: torch.device
    model_name: str
    num_frames: int
    minimum_frames: int
    image_size_hw: tuple[int, int] | None


def _model_state_dict(checkpoint: Mapping[str, Any], *, location: str) -> dict[str, Any]:
    """Extract only the explicit ``model.`` weights and reject empty payloads."""
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint has no state_dict mapping."
        )
    weights = {
        str(key)[len(MODEL_STATE_PREFIX) :]: value
        for key, value in raw_state.items()
        if isinstance(key, str) and key.startswith(MODEL_STATE_PREFIX)
    }
    if not weights:
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint state_dict carries no "
            f"{MODEL_STATE_PREFIX!r} model weights."
        )
    return weights


def _restore_model_weights(
    model: nn.Module, checkpoint: Mapping[str, Any], *, location: str
) -> None:
    """Strictly restore the model parameters and buffers from one checkpoint."""
    weights = _model_state_dict(checkpoint, location=location)
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise BallInferenceCheckpointError(
            f"{location}: model state_dict does not match the architecture: {error}"
        ) from error


def _required_image_size(config: DictConfig, *, location: str) -> tuple[int, int]:
    container = cast(Mapping[str, Any], config)
    model_block = container.get("model")
    if not isinstance(model_block, Mapping):
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint config has no model mapping."
        )
    if str(model_block.get("name")) == "dinov3_rope":
        source: Any = model_block.get("image_size")
    else:
        data_block = container.get("data")
        source = data_block.get("image_size") if isinstance(data_block, Mapping) else None
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence) or len(source) != 2:
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint config does not declare a 2-element "
            "image_size for the configured model."
        )
    height, width = (value for value in source)
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint image_size must be two positive integers, "
            f"got {list(source)!r}."
        )
    return height, width


def load_ball_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> LoadedBallModel:
    """Build the checkpoint's architecture and restore its weights strictly."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    location = f"Ball detection checkpoint {path}"

    config = read_checkpoint_config(path)
    bound = build_ball_detection_pair(config)
    adapter = bound.adapter
    if not isinstance(adapter, BallModelIOAdapter):
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint does not contain a ball model-I/O adapter."
        )
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(raw, Mapping):
        raise BallInferenceCheckpointError(f"{location}: checkpoint root must be a mapping.")
    _restore_model_weights(bound.model, cast(Mapping[str, Any], raw), location=location)

    model_name = adapter.spec.model_name
    minimum_frames = MINIMUM_FRAMES_BY_MODEL.get(model_name, adapter.minimum_frames)
    if adapter.minimum_frames != minimum_frames:
        raise BallInferenceCheckpointError(
            f"{location}: adapter minimum_frames={adapter.minimum_frames} disagrees "
            f"with the review catalog value {minimum_frames} for {model_name!r}."
        )
    num_frames = adapter.spec.configured_frames
    if num_frames < minimum_frames:
        raise BallInferenceCheckpointError(
            f"{location}: {model_name} requires at least {minimum_frames} frames, "
            f"but the checkpoint declares model.num_frames={num_frames}."
        )

    resolved = resolve_device(device)
    bound.model.to(resolved)
    bound.model.eval()
    return LoadedBallModel(
        model=bound.model,
        adapter=adapter,
        device=resolved,
        model_name=model_name,
        num_frames=num_frames,
        minimum_frames=minimum_frames,
        image_size_hw=_required_image_size(config, location=location),
    )


__all__ = [
    "MODEL_STATE_PREFIX",
    "BallInferenceCheckpointError",
    "LoadedBallModel",
    "load_ball_model",
]
