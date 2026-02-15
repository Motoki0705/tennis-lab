"""Model components and factory for ball multi-task learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.ball_multitask.models.backbone import BallMultitaskBackbone
from src.ball_multitask.models.multitask_model import BallMultitaskModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_ball_multitask_model(config: DictConfig) -> nn.Module:
    """Build a ball multitask model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "multitask_transformer"))
    if model_name == "multitask_transformer":
        return BallMultitaskModel.from_config(config)
    raise ValueError(
        "Unknown ball_multitask model.name="
        f"'{model_name}'. Supported: ['multitask_transformer']"
    )


__all__ = [
    "BallMultitaskBackbone",
    "BallMultitaskModel",
    "build_ball_multitask_model",
]
