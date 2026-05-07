"""Models and factory for UV trajectory completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.trajectory_completion.models.discriminators import (
    build_trajectory_completion_discriminator,
)
from src.tasks.trajectory_completion.models.uv_completion_model import (
    UVTrajectoryCompletionModel,
)
from src.tasks.trajectory_completion.models.uv_completion_nocourt_model import (
    UVTrajectoryCompletionNoCourtModel,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_trajectory_completion_model(config: DictConfig) -> nn.Module:
    """Build a trajectory completion model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "uv_transformer"))
    if model_name == "uv_transformer":
        return UVTrajectoryCompletionModel.from_config(config)
    if model_name == "uv_transformer_nocourt":
        return UVTrajectoryCompletionNoCourtModel.from_config(config)
    raise ValueError(
        "Unknown trajectory_completion model.name="
        f"'{model_name}'. Supported: ['uv_transformer', 'uv_transformer_nocourt']"
    )


__all__ = [
    "UVTrajectoryCompletionModel",
    "UVTrajectoryCompletionNoCourtModel",
    "build_trajectory_completion_discriminator",
    "build_trajectory_completion_model",
]
