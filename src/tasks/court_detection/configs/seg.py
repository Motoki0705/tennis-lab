"""Court cell segmentation configuration.

Defines hyper-parameters and paths for training a court cell
segmentation model (6 in-court cells + background = 7 classes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

COURT_DATA_DIR = _REPO_ROOT / "data" / "court"
OUTPUT_DIR = _REPO_ROOT / "outputs" / "court_detection" / "seg"
CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "court_detection" / "seg"

CELL_NAMES: tuple[str, ...] = (
    "background",
    "left_svc_box",
    "right_svc_box",
    "left_back_court",
    "right_back_court",
    "left_doubles_alley",
    "right_doubles_alley",
)

NUM_SEG_CLASSES: int = len(CELL_NAMES)  # 7

SEG_HFLIP_SWAP: list[tuple[int, int]] = [(1, 2), (3, 4), (5, 6)]


@dataclass
class CourtSegConfig:
    """Hyper-parameters for court cell segmentation training."""

    in_channels: int = 3
    num_classes: int = NUM_SEG_CLASSES

    train_scales: list[int] = field(default_factory=lambda: [288])
    val_short_side: int = 288

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
    grad_clip_max_norm: float = 1.0

    ce_weight: float = 1.0
    dice_weight: float = 1.0

    seed: int = 42
    dry_run_epochs: int = 2
    dry_run_batch_size: int = 2
