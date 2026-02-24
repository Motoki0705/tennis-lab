"""Court detection models and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.court_detection.models.court_keypoint_model import CourtKeypointModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def build_court_detection_model(config: DictConfig) -> nn.Module:
    """Build a court detection model from ``config.model.name``."""
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "vit_heatmap"))
    if model_name == "vit_heatmap":
        return CourtKeypointModel.from_config(config)
    raise ValueError(
        "Unknown court_detection model.name="
        f"'{model_name}'. Supported: ['vit_heatmap']"
    )


__all__ = ["CourtKeypointModel", "build_court_detection_model"]
