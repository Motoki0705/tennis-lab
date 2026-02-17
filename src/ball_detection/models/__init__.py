"""Model factory for ball_detection."""

from __future__ import annotations

from typing import Any

from src.ball_detection.models.ball_detector_model import BallDetectorModel


def build_model(config: Any | None = None) -> BallDetectorModel:
    """Build the temporal memory-attention detector from config."""
    return BallDetectorModel.from_config(config or {})


__all__ = ["BallDetectorModel", "build_model"]
