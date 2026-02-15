"""Models and factory for UV trajectory completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.trajectory_completion.models.uv_completion_model import (
    UVTrajectoryCompletionModel,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_trajectory_completion_model(config: DictConfig) -> nn.Module:
    """Build a trajectory completion model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "uv_transformer"))
    if model_name == "uv_transformer":
        return UVTrajectoryCompletionModel.from_config(config)
    raise ValueError(
        "Unknown trajectory_completion model.name="
        f"'{model_name}'. Supported: ['uv_transformer']"
    )


__all__ = ["UVTrajectoryCompletionModel", "build_trajectory_completion_model"]
