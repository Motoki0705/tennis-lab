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
from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.models.query_encoder.contracts import (
    CourtQueryRawOutput,
    PatchTokenBatch,
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
class CourtQueryModelSpec:
    """Static raw-output contract for the additive query model."""

    target_bundle: CourtTargetBundleSpec
    in_channels: int
    short_side: int
    model_family: Literal["court_query_encoder"] = "court_query_encoder"


@dataclass(frozen=True, slots=True)
class CourtModelCall:
    images: Tensor
    model_args: tuple[Tensor, ...]
    batch_size: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class CourtQueryModelCall:
    images: Tensor
    patch_batch: PatchTokenBatch
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
class CourtPoseTargetBatch:
    translation_m: Tensor
    rotation: Tensor
    log_focal: Tensor
    intrinsics: Tensor
    semantic_to_physical: Tensor
    raw_pose10d: Tensor


@dataclass(frozen=True, slots=True)
class CourtQueryTrainingCall:
    model_call: CourtQueryModelCall
    dense_targets: Mapping[CourtTargetKind, object]
    pose_target: CourtPoseTargetBatch
    batch: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CourtQueryTrainingResult:
    loss: Tensor
    dense_losses: Mapping[CourtTargetKind, Tensor]
    pose_losses: Mapping[str, Tensor]
    output: CourtQueryRawOutput
    decoded_pose: CourtDecodedPose


@dataclass(frozen=True, slots=True)
class CourtQueryDecodedOutput:
    pose: CourtDecodedPose
    dense_logits: CourtLogits


@dataclass(frozen=True, slots=True)
class CourtQueryPrediction:
    """Typed prediction payload before persistence flattening."""

    pose: CourtDecodedPose
    dense: Mapping[CourtTargetKind, object]


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
    "CourtQueryModelCall",
    "CourtQueryDecodedOutput",
    "CourtQueryModelSpec",
    "CourtQueryPrediction",
    "CourtQueryTrainingCall",
    "CourtQueryTrainingResult",
    "CourtQueryRawOutput",
    "CourtSegmentationPrediction",
    "CourtPoseTargetBatch",
    "CourtTrainingCall",
    "CourtTrainingResult",
]
