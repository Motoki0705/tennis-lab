"""Types for ball_detection inference configuration and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class InferenceMemberConfig:
    """One predictor member for single/ensemble inference."""

    backend: str
    checkpoint: Path
    weight: float
    score_threshold: float


@dataclass(frozen=True)
class InferenceConfig:
    """Resolved runtime config for inference."""

    strategy: str
    device: str
    image_h: int
    image_w: int
    batch_size: int
    max_frames: int | None
    window_size: int | None
    clip_frames: int | None
    clip_stride: int | None
    visibility_threshold: float
    single_member: InferenceMemberConfig
    ensemble_members: tuple[InferenceMemberConfig, ...]


@dataclass(frozen=True)
class InferenceResult:
    """Per-frame 2D ball predictions."""

    frame_indices: IntArray
    ball_uv: FloatArray
    ball_xy_px: FloatArray
    visibility: BoolArray
    score: FloatArray
