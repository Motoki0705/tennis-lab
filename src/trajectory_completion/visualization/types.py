"""Types for trajectory completion visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for trajectory completion visualization."""

    mode: str
    scene_path: Path
    camera: Any
    frame: int
    view: str
    fps: float | None
    save: Path | None
    info: bool
    checkpoint: str | None
    device: str
    output: str | None
    seed: int
    apply_corruption: bool
    use_scene_visibility: bool
    connector_stride: int
    start: int
    max_frames: int | None
    error_threshold: float
    show_court_lines: bool
    hydra_cfg: DictConfig


@dataclass(frozen=True)
class TrajectoryInputs:
    """Prepared UV trajectory inputs for visualization and inference."""

    ball_uv_gt: np.ndarray
    ball_uv_in: np.ndarray
    ball_gt_visible: np.ndarray
    ball_obs_mask: np.ndarray
    court_kp: np.ndarray
    court_vis: np.ndarray
    meta: dict[str, Any]
    camera_idx: int
    start: int
