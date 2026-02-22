"""Types for ball_detection visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from src.ball_detection.inference.types import InferenceConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for ball-detection visualization."""

    mode: str
    video_path: Path
    output_video_path: Path | None
    output_npz_path: Path | None
    fps: float | None
    info: bool
    radius: int
    thickness: int
    color_detected_bgr: tuple[int, int, int]
    color_trail_bgr: tuple[int, int, int]
    show_score: bool
    show_trail: bool
    trail_length: int
    inference: InferenceConfig
    hydra_cfg: DictConfig


@dataclass(frozen=True)
class VideoInputs:
    """Decoded video frames and metadata for visualization/prediction."""

    frames_rgb: NDArray[np.uint8]
    width: int
    height: int
    fps: float
