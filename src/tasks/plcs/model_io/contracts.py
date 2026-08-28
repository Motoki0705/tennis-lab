"""Typed PLCS model input, output, and inference contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from torch import Tensor

from src.tasks.base.model_io import ModelCall


class PLCSInputProfile(StrEnum):
    """Resolved PLCS tensor layout selected at composition time."""

    FRAME = "frame"
    SEQUENCE = "sequence"
    MULTIVIEW = "multiview"
    TRACK_QUERY = "track_query"


@dataclass(frozen=True, slots=True)
class PLCSReprojectionTarget:
    """Clean 2D pose targets and fixed cameras for reprojection supervision."""

    target_uv: Tensor
    target_vis: Tensor
    padding_mask: Tensor
    camera_R: Tensor
    camera_C: Tensor
    camera_f: Tensor
    camera_cx: Tensor
    camera_cy: Tensor
    camera_w: Tensor
    camera_h: Tensor


@dataclass(frozen=True, slots=True)
class PLCSPreparedBatch:
    """Validated model call plus the output layout required by its consumer."""

    call: ModelCall
    sequence_shape: tuple[int, int] | None = None
    target_position: Tensor | None = None
    target_rotation: Tensor | None = None
    target_human_kp_3d: Tensor | None = None
    target_padding_mask: Tensor | None = None
    reprojection_target: PLCSReprojectionTarget | None = None


@dataclass(frozen=True, slots=True)
class PLCSDecodedPrediction:
    """Canonical decoded PLCS model output."""

    position: Tensor
    rotation: Tensor
    canonical_pose: Tensor | None = None
    auxiliary_position: Tensor | None = None


@dataclass(frozen=True, slots=True)
class PLCSTrackingDecodedPrediction:
    """Canonical decoded output for the fixed track-query profile."""

    position: Tensor
    rotation: Tensor
    presence_logits: Tensor


Float32Array: TypeAlias = np.ndarray


@dataclass(frozen=True, slots=True)
class PLCSPhysicalPrediction:
    """CPU NumPy prediction used by integrated inference consumers."""

    position_meters: Float32Array
    yaw_radians: Float32Array
    canonical_pose: Float32Array | None = None


__all__ = [
    "Float32Array",
    "PLCSDecodedPrediction",
    "PLCSInputProfile",
    "PLCSPhysicalPrediction",
    "PLCSPreparedBatch",
    "PLCSReprojectionTarget",
    "PLCSTrackingDecodedPrediction",
]
