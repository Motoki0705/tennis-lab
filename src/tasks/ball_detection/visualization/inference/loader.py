"""Review UI metadata around the shared inference-only ball checkpoint loader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch import nn

from src.tasks.ball_detection.inference.checkpoint import (
    BallInferenceCheckpointError,
    load_ball_checkpoint,
)
from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.normalization import (
    IDENTITY_NORMALIZATION,
    BallImageNormalization,
)
from src.tasks.ball_detection.visualization.review.checkpoints import (
    MINIMUM_FRAMES_BY_MODEL,
)
from src.utils.device import resolve_device


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
    image_normalization: BallImageNormalization = IDENTITY_NORMALIZATION


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

    loaded = load_ball_checkpoint(path)
    config, bound = loaded.config, loaded.model_io
    adapter = bound.adapter
    if not isinstance(adapter, BallModelIOAdapter):
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint does not contain a ball model-I/O adapter."
        )
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
        image_normalization=loaded.image_normalization,
    )


__all__ = [
    "LoadedBallModel",
    "load_ball_model",
]
