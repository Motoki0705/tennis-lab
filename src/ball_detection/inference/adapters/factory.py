"""Factory for selecting model-specific temporal input adapters."""

from __future__ import annotations

from torch import nn

from src.ball_detection.inference.adapters.base import ModelInputAdapter
from src.ball_detection.inference.adapters.hrnet import HRNetContextInputAdapter
from src.ball_detection.inference.adapters.tracknetv3 import TrackNetV3InputAdapter


def build_adapter_for_model(model: nn.Module) -> ModelInputAdapter:
    """Infer an adapter from model attributes."""
    seq_len = getattr(model, "seq_len", None)
    if seq_len is not None:
        return TrackNetV3InputAdapter(seq_len=int(seq_len))

    backbone = getattr(model, "backbone", None)
    input_channels = getattr(backbone, "input_channels", None)
    if input_channels is not None:
        channels = int(input_channels)
        if channels % 3 != 0:
            raise ValueError(f"HRNet input channels must be divisible by 3, got {channels}")
        return HRNetContextInputAdapter(context_frames=channels // 3)

    raise ValueError(
        "Could not infer input adapter for model. "
        "Provide an explicit adapter implementation."
    )
