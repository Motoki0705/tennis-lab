"""Typed bundle model-I/O contracts for Court detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from torch import Tensor

from src.tasks.base.model_io import ModelIOContractError
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
)

CourtEncoderKind = Literal["default", "dinov3"]
CourtLogits: TypeAlias = Mapping[CourtTargetKind, Tensor]


class CourtModelIOError(ModelIOContractError):
    """Raised when a Court model, batch, or output violates its contract."""


@dataclass(frozen=True, slots=True)
class CourtModelSpec:
    """Static model/preprocessing contract selected at composition."""

    target_bundle: CourtTargetBundleSpec
    in_channels: int
    short_side: int
    encoder_kind: CourtEncoderKind = "default"


@dataclass(frozen=True, slots=True)
class CourtModelCall:
    images: Tensor
    model_args: tuple[Tensor, ...]
    batch_size: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class CourtTrainingCall:
    model_call: CourtModelCall
    targets: Mapping[CourtTargetKind, object]
    batch: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CourtTrainingResult:
    loss: Tensor
    losses: Mapping[CourtTargetKind, Tensor]
    logits: CourtLogits


@dataclass(frozen=True, slots=True)
class CourtKeypointPrediction:
    """Decoded one- or multi-peak channels in original-image pixels."""

    keypoints: Tensor  # [C,P,2]
    scores: Tensor  # [C,P]
    valid: Tensor  # [C,P]
    heatmaps: Tensor  # [C,H,W]


@dataclass(frozen=True, slots=True)
class CourtSegmentationPrediction:
    mask: Tensor
    logits: Tensor


@dataclass(frozen=True, slots=True)
class CourtLinePrediction:
    probability: Tensor
    logits: Tensor


CourtDecodedPrediction: TypeAlias = (
    CourtKeypointPrediction | CourtSegmentationPrediction | CourtLinePrediction
)


__all__ = [
    "CourtDecodedPrediction",
    "CourtEncoderKind",
    "CourtKeypointPrediction",
    "CourtLinePrediction",
    "CourtLogits",
    "CourtModelCall",
    "CourtModelIOError",
    "CourtModelSpec",
    "CourtSegmentationPrediction",
    "CourtTrainingCall",
    "CourtTrainingResult",
]
