"""Types for event detection visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for event visualization."""

    task: str
    mode: str
    scene_path: Path
    camera: Any
    fps: float
    save: Path | None
    info: bool
    checkpoint: str | None
    device: str
    output: str | None
    seed: int
    threshold: float
    min_distance: int
    top_k: int | None
    event_radius_frames: int
    event_sigma_frames: float
    show_court_lines: bool
    hydra_cfg: DictConfig


@dataclass(frozen=True)
class UVEventInputs:
    """Loaded UV inputs and GT labels for a single scene."""

    ball_uv: np.ndarray
    ball_vis: np.ndarray
    court_kp: np.ndarray
    court_vis: np.ndarray
    targets: np.ndarray
    shot_indices: list[int]
    bounce_indices: list[int]
    meta: dict[str, Any]
    camera_idx: int


@dataclass(frozen=True)
class Traj3DEventInputs:
    """Loaded 3D trajectory inputs and GT labels for a single scene."""

    ball_pos_world: np.ndarray
    targets: np.ndarray
    shot_indices: list[int]
    bounce_indices: list[int]
    meta: dict[str, Any]
