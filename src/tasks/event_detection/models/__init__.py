"""Model definitions and factory for event detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.event_detection.models.discriminators import (
    build_event_detection_discriminator,
)
from src.tasks.event_detection.models.traj3d_event_model import Traj3DEventModel
from src.tasks.event_detection.models.uv_event_model import UVEventModel
from src.tasks.event_detection.models.uv_event_nocourt_model import UVEventNoCourtModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_event_detection_model(config: DictConfig) -> nn.Module:
    """Build an event detection model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "uv_transformer"))
    if model_name == "uv_transformer":
        return UVEventModel.from_config(config)
    if model_name == "uv_transformer_nocourt":
        return UVEventNoCourtModel.from_config(config)
    if model_name == "traj3d_transformer":
        return Traj3DEventModel.from_config(config)
    raise ValueError(
        "Unknown event_detection model.name="
        f"'{model_name}'. Supported: ['uv_transformer', 'uv_transformer_nocourt', 'traj3d_transformer']"
    )


__all__ = [
    "UVEventModel",
    "UVEventNoCourtModel",
    "Traj3DEventModel",
    "build_event_detection_discriminator",
    "build_event_detection_model",
]
