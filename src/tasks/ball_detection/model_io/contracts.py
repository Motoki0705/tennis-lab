"""Typed model-I/O contracts for ball detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

from src.tasks.base.model_io import ModelIOContractError

BallInputMode = Literal["rgb", "mdd"]
BallInputLayout = Literal["bcthw", "btchw"]


class BallModelIOError(ModelIOContractError):
    """Raised when a ball model, input, or output violates its I/O contract."""


@dataclass(frozen=True)
class BallModelInputSpec:
    """Static input contract resolved once from the selected model config."""

    model_name: str
    input_mode: BallInputMode
    input_layout: BallInputLayout
    in_channels: int
    num_classes: int
    configured_frames: int
    image_size_hw: tuple[int, int] | None
    minimum_spatial_size: int | None
    mdd_gain: float
    mdd_offset: float


@dataclass(frozen=True)
class BallModelCall:
    """Validated tensor call for one ball-detector forward pass."""

    images: Tensor
    model_input: Tensor
    model_args: tuple[Tensor, ...]
    batch_size: int
    frame_count: int


@dataclass(frozen=True)
class BallTrainingCall:
    """Validated training batch plus the prepared model call."""

    model_call: BallModelCall
    target_heatmaps: Tensor
    coords: Tensor
    visibility: Tensor
    original_size: Tensor


@dataclass(frozen=True)
class BallPrediction:
    """Decoded inference result with stable typed fields."""

    coords: Tensor
    confidence: Tensor
    heatmaps: Tensor


class BallHeatmapPredictor(Protocol):
    """Evaluation boundary for probability heatmap prediction."""

    @property
    def device(self) -> torch.device:
        """Device on which prediction is executed."""
        ...

    def predict_heatmaps(
        self,
        images: Tensor,
        *,
        target_size_hw: tuple[int, int],
    ) -> Tensor:
        """Return probability heatmaps with shape ``(B, T, H, W)``."""
        ...


__all__ = [
    "BallHeatmapPredictor",
    "BallInputLayout",
    "BallInputMode",
    "BallModelCall",
    "BallModelIOError",
    "BallModelInputSpec",
    "BallPrediction",
    "BallTrainingCall",
]
