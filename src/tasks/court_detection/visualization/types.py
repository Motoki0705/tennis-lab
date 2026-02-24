"""Types for court detection visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings for court visualization."""

    mode: str
    input_path: Path
    output_dir: Path
    num_samples: int
    checkpoint: Path | None
    device: str
    save_overlay: bool
    save_json: bool
    point_radius: int
    point_color: tuple[int, int, int]
    line_color: tuple[int, int, int]
    text_color: tuple[int, int, int]
    line_thickness: int
    show_keypoint_ids: bool
    show_court_lines: bool
    visibility_threshold: float
    hydra_cfg: DictConfig


@dataclass(frozen=True)
class SceneImage:
    """Loaded image input for prediction/visualization."""

    image_path: Path
    image_rgb: np.ndarray


@dataclass(frozen=True)
class KeypointPrediction:
    """Prediction outputs for a single image."""

    keypoints: np.ndarray
    visibility: np.ndarray


@dataclass(frozen=True)
class RunSummary:
    """Execution summary for reporting."""

    total_inputs: int
    succeeded: int
    failed: list[Path]
