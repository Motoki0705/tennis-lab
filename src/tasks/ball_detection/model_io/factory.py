"""Composition factory for verified ball model/I/O pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from src.tasks.ball_detection.model_io.adapters import (
    BallModelIOAdapter,
    build_ball_model_input_spec,
)
from src.tasks.ball_detection.models.conv_next_unet import ConvNeXtUNet
from src.tasks.base.model_io import BoundModelIO, bind_model_io

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_ball_detection_pair(
    config: DictConfig,
) -> BoundModelIO[Tensor, Tensor, Tensor]:
    """Select and verify one matching ball model/adapter pair."""
    spec = build_ball_model_input_spec(config)
    model = ConvNeXtUNet.from_config(config)
    adapter = BallModelIOAdapter(spec, expected_model_type=ConvNeXtUNet, minimum_frames=1)
    adapter.validate_model_pair(model)
    return bind_model_io(model, adapter)


__all__ = ["build_ball_detection_pair"]
