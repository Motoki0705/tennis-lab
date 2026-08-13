"""Typed model-I/O contracts for court keypoint, segmentation, and line tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from src.tasks.base.model_io import ModelIOContractError

CourtTask = Literal["kp", "seg", "line"]
CourtEncoderKind = Literal["default", "dinov3"]


class CourtModelIOError(ModelIOContractError):
    """Raised when a court model, batch, or output violates its contract."""


@dataclass(frozen=True, slots=True)
class CourtModelSpec:
    """Static model and preprocessing contract selected at composition."""

    task: CourtTask
    in_channels: int
    output_channels: int
    short_side: int
    encoder_kind: CourtEncoderKind = "default"


@dataclass(frozen=True, slots=True)
class CourtModelCall:
    """Validated court model input."""

    images: Tensor
    model_args: tuple[Tensor, ...]
    batch_size: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class CourtTrainingCall:
    """Validated task-specific training tensors."""

    model_call: CourtModelCall
    target: Tensor
    batch: dict[str, object]


@dataclass(frozen=True, slots=True)
class CourtTrainingResult:
    """Canonical training result decoded by a task adapter."""

    loss: Tensor
    logits: Tensor


@dataclass(frozen=True, slots=True)
class CourtKeypointPrediction:
    """Decoded one- or multi-peak channels in original-image pixels."""

    keypoints: Tensor  # [C,P,2]
    scores: Tensor  # [C,P]
    valid: Tensor  # [C,P]
    covariance: Tensor  # [C,P,2,2], original-image pixel coordinates
    heatmaps: Tensor  # [C,H,W]
    semantic_class_names: tuple[str, ...] | None = None
    image_size_hw: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class CourtSegmentationPrediction:
    """Decoded multi-class court segmentation output."""

    mask: Tensor
    logits: Tensor


@dataclass(frozen=True, slots=True)
class CourtLinePrediction:
    """Decoded binary court-line probability output."""

    probability: Tensor
    logits: Tensor


__all__ = [
    "CourtEncoderKind",
    "CourtKeypointPrediction",
    "CourtLinePrediction",
    "CourtModelCall",
    "CourtModelIOError",
    "CourtModelSpec",
    "CourtSegmentationPrediction",
    "CourtTask",
    "CourtTrainingCall",
    "CourtTrainingResult",
]
