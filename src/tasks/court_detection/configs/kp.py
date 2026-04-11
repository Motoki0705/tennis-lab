"""Court keypoint heatmap configuration.

Defines hyper-parameters and paths for training a court keypoint
heatmap regression model (14 keypoint channels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

COURT_DATA_DIR = _REPO_ROOT / "data" / "court"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "court_detection" / "kp"
CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "court_detection" / "kp"

NUM_KEYPOINTS: int = 14

KP_HFLIP_SWAP: list[tuple[int, int]] = [
    (0, 1), (2, 3), (4, 6), (5, 7), (8, 9), (10, 11),
]


@dataclass
class CourtKPConfig:
    """Hyper-parameters for court keypoint heatmap training."""

    in_channels: int = 3
    num_classes: int = NUM_KEYPOINTS

    train_scales: list[int] = field(default_factory=lambda: [288])
    val_short_side: int = 288

    gaussian_sigma: float = 3.0

    crop_scale: tuple[float, float] = (0.2, 1.0)
    crop_ratio: tuple[float, float] = (0.5, 2.0)
    hflip_prob: float = 0.7
    affine_degrees: float = 25.0
    affine_translate: tuple[float, float] = (0.18, 0.18)
    affine_scale: tuple[float, float] = (0.65, 1.5)
    affine_shear: float = 18.0
    perspective_distortion: float = 0.25
    perspective_prob: float = 0.6
    color_jitter: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.2)
    gaussian_blur_kernel: list[int] = field(default_factory=lambda: [3, 5, 7, 9])
    gaussian_blur_sigma: tuple[float, float] = (0.1, 3.0)
    gaussian_blur_prob: float = 0.5

    batch_size: int = 8
    num_workers: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    grad_clip_max_norm: float = 1.0

    seed: int = 42
    dry_run_epochs: int = 2
    dry_run_batch_size: int = 2
