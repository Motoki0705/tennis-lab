"""Types for ball multitask visualization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for prediction/visualization."""

    mode: str
    scene_path: Path
    camera: Any
    checkpoint: Path | None
    device: str
    output: Path | None
    save_dir: Path | None
    save_format: str
    renderers: tuple[str, ...]
    fps: float
    event_radius_frames: int
    event_sigma_frames: float
    show_court_lines: bool
    info: bool
    threshold: float
    min_distance: int
    top_k: int | None
    denormalize: bool
    hydra_cfg: DictConfig


@dataclass(frozen=True)
class SceneInputs:
    """Scene arrays loaded from NPZ before model adaptation."""

    ball_uv: np.ndarray
    ball_vis: np.ndarray
    court_kp: np.ndarray
    court_vis: np.ndarray
    ball_pos_world: np.ndarray | None
    seq_len: int
    meta: dict[str, Any]
    camera_idx: int
