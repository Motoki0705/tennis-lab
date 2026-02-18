"""Model factory for ball_detection."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.ball_detection.models.ball_detector_model import BallDetectorModel
from src.ball_detection.models.hrnet_temporal_heatmap_model import WASBHRNetTemporalModel
from src.ball_detection.models.tracknetv3_heatmap_model import TrackNetV3HeatmapModel


def build_model(config: Any | None = None) -> nn.Module:
    """Build a ball_detection model from config."""
    cfg = config or {}
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    model_name = str(model_cfg.get("name", "temporal_memory_ball_detector")).lower()

    if model_name == "temporal_memory_ball_detector":
        return BallDetectorModel.from_config(cfg)
    if model_name in {"wasb_hrnet_temporal", "hrnet_temporal_heatmap"}:
        return WASBHRNetTemporalModel.from_config(cfg)
    if model_name in {"tracknetv3", "tracknetv3_heatmap"}:
        return TrackNetV3HeatmapModel.from_config(cfg)

    raise ValueError(f"Unsupported ball_detection model name: {model_name}")


__all__ = [
    "BallDetectorModel",
    "TrackNetV3HeatmapModel",
    "WASBHRNetTemporalModel",
    "build_model",
]
