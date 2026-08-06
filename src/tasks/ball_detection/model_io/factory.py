"""Composition factory for verified ball model/I/O pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from src.tasks.ball_detection.model_io.adapters import (
    BallModelIOAdapter,
    DINOv3BallExecutionBoundary,
    build_ball_model_input_spec,
)
from src.tasks.ball_detection.model_io.contracts import BallModelIOError
from src.tasks.ball_detection.models.conv_next_unet import ConvNeXtUNet
from src.tasks.ball_detection.models.dinov3_rope import DINOv3RoPEBallDetector
from src.tasks.ball_detection.models.spatiotemporal_unet import SpatioTemporalUNet
from src.tasks.base.model_io import BoundModelIO, bind_model_io

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_ball_detection_pair(
    config: DictConfig,
) -> BoundModelIO[Tensor, Tensor, Tensor]:
    """Select and verify one matching ball model/adapter pair."""
    spec = build_ball_model_input_spec(config)
    model: nn.Module
    if spec.model_name == "stunet":
        model = SpatioTemporalUNet.from_config(config)
        adapter = BallModelIOAdapter(
            spec,
            expected_model_type=SpatioTemporalUNet,
            minimum_frames=8,
        )
    elif spec.model_name == "conv_next_unet":
        model = ConvNeXtUNet.from_config(config)
        adapter = BallModelIOAdapter(
            spec,
            expected_model_type=ConvNeXtUNet,
            minimum_frames=1,
        )
    elif spec.model_name == "dinov3_rope":
        model = DINOv3RoPEBallDetector.from_config(config)
        adapter = BallModelIOAdapter(
            spec,
            expected_model_type=DINOv3RoPEBallDetector,
            minimum_frames=1,
            execution_boundary=DINOv3BallExecutionBoundary(
                frozen_backbone=(
                    model.backbone_train_mode == "frozen"
                    and not model.backbone_lora_enabled
                )
            ),
        )
    else:
        raise BallModelIOError(
            f"Unsupported ball model.name={spec.model_name!r}; expected one of "
            "['stunet', 'conv_next_unet', 'dinov3_rope']."
        )
    adapter.validate_model_pair(model)
    return bind_model_io(model, adapter)


__all__ = ["build_ball_detection_pair"]
