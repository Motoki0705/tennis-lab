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
    supervised: Tensor
    original_size: Tensor


@dataclass(frozen=True)
class BallCandidateConfig:
    """Threshold-free local-peak evidence; sizes are in native heatmap cells."""

    max_candidates: int = 8
    nms_kernel: int = 5
    patch_size: int = 5

    def __post_init__(self) -> None:
        for name in ("max_candidates", "nms_kernel", "patch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.nms_kernel % 2 == 0 or self.patch_size % 2 == 0:
            raise ValueError("nms_kernel and patch_size must be odd.")


DEFAULT_CANDIDATE_CONFIG = BallCandidateConfig()


@dataclass(frozen=True)
class BallCandidates:
    """Unthresholded peaks, padded to K with explicit masks.

    coords: (B,T,K,2), x/(W-1), y/(H-1), optionally subpixel refined.
    scores: (B,T,K), native sigmoid cell values, not existence probabilities.
    valid: (B,T,K), a contrastive local peak exists (not an observation gate).
    cells: (B,T,K,2), integer x,y lattice centres before subpixel refinement.
    patches: (B,T,K,P,P), native probability values around those centres.
    patch_valid: (B,T,K,P,P), in-map cells of a valid candidate.
    Invalid slots and out-of-map patch values are zero, never fabricated peaks.
    """

    coords: Tensor
    scores: Tensor
    valid: Tensor
    cells: Tensor
    patches: Tensor
    patch_valid: Tensor
    config: BallCandidateConfig


@dataclass(frozen=True)
class BallPrediction:
    """Decoded inference result with stable typed fields."""

    coords: Tensor
    confidence: Tensor
    heatmaps: Tensor
    candidates: BallCandidates


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
    "BallCandidateConfig",
    "BallCandidates",
    "BallHeatmapPredictor",
    "BallInputLayout",
    "BallInputMode",
    "BallModelCall",
    "BallModelIOError",
    "BallModelInputSpec",
    "BallPrediction",
    "BallTrainingCall",
]
