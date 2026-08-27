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
from src.tasks.court_detection.models.pose_head import (
    CourtModelOutput,
    CourtRawPoseOutput,
)

CourtEncoderKind = Literal["default", "dinov3"]
CourtLogits: TypeAlias = Mapping[CourtTargetKind, Tensor]
CourtTrainingTargetKind: TypeAlias = CourtTargetKind | Literal["pose", "image_size"]
CourtPoseLossKind: TypeAlias = Literal[
    "pose_translation",
    "pose_rotation",
    "pose_focal",
]


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
    targets: Mapping[CourtTrainingTargetKind, object]
    batch: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CourtTrainingResult:
    """Dense-only loss result with backward-compatible weighted fields.

    ``loss`` and ``losses`` retain the original weighted objective contract.
    The additional maps make the unweighted term and its configured/effective
    outer weight explicit for experiment accounting.
    """

    loss: Tensor
    losses: Mapping[CourtTargetKind, Tensor]
    logits: CourtLogits
    raw_loss: Tensor
    raw_losses: Mapping[CourtTargetKind, Tensor]
    configured_weights: Mapping[CourtTargetKind, Tensor]
    effective_weights: Mapping[CourtTargetKind, Tensor]
    weighted_losses: Mapping[CourtTargetKind, Tensor]


@dataclass(frozen=True, slots=True)
class CourtPoseTargetBatch:
    translation_m: Tensor
    rotation: Tensor
    log_focal: Tensor
    intrinsics: Tensor
    semantic_to_physical: Tensor
    raw_pose10d: Tensor


@dataclass(frozen=True, slots=True)
class CourtConsistencyResult:
    """Typed auxiliary objective, schedule weights, and geometry diagnostics."""

    coordinate_loss: Tensor
    cheirality_loss: Tensor
    auxiliary_loss: Tensor
    weighted_auxiliary_loss: Tensor
    configured_weight: Tensor
    effective_weight: Tensor
    visible_point_count: Tensor
    mean_distance_px: Tensor
    invalid_depth_rate: Tensor
    dense_points_xy: Tensor
    pose_points_xy: Tensor
    pose_depth_m: Tensor


@dataclass(frozen=True, slots=True)
class CourtPoseTrainingResult:
    """Typed raw/configured/effective/weighted pose loss decomposition."""

    loss: Tensor
    raw_dense_loss: Tensor
    direct_dense_loss: Tensor
    direct_pose_loss: Tensor
    raw_dense_losses: Mapping[CourtTargetKind, Tensor]
    dense_losses: Mapping[CourtTargetKind, Tensor]
    dense_configured_weights: Mapping[CourtTargetKind, Tensor]
    dense_effective_weights: Mapping[CourtTargetKind, Tensor]
    weighted_dense_losses: Mapping[CourtTargetKind, Tensor]
    pose_losses: Mapping[CourtPoseLossKind, Tensor]
    weighted_pose_losses: Mapping[CourtPoseLossKind, Tensor]
    pose_configured_weights: Mapping[CourtPoseLossKind, Tensor]
    pose_effective_weights: Mapping[CourtPoseLossKind, Tensor]
    consistency: CourtConsistencyResult | None
    output: CourtModelOutput
    decoded_pose: CourtDecodedPose


@dataclass(frozen=True, slots=True)
class CourtDecodedOutput:
    pose: CourtDecodedPose
    dense_logits: CourtLogits


@dataclass(frozen=True, slots=True)
class CourtPosePrediction:
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

CourtRawOutput = CourtModelOutput
CourtHierarchicalTrainingResult = CourtPoseTrainingResult
CourtModelTrainingResult = CourtPoseTrainingResult
# Read-only source aliases for older metric/checkpoint consumers.  They point
# at the unified contracts and do not restore a separate model branch.
CourtQueryConsistencyResult = CourtConsistencyResult


__all__ = [
    "CourtDecodedPrediction",
    "CourtEncoderKind",
    "CourtKeypointPrediction",
    "CourtLinePrediction",
    "CourtLogits",
    "CourtModelCall",
    "CourtModelIOError",
    "CourtModelSpec",
    "CourtConsistencyResult",
    "CourtQueryConsistencyResult",
    "CourtSegmentationPrediction",
    "CourtPoseTargetBatch",
    "CourtPoseLossKind",
    "CourtTrainingCall",
    "CourtTrainingTargetKind",
    "CourtTrainingResult",
    "CourtModelOutput",
    "CourtRawPoseOutput",
    "CourtPoseTrainingResult",
    "CourtHierarchicalTrainingResult",
    "CourtModelTrainingResult",
    "CourtRawOutput",
]
