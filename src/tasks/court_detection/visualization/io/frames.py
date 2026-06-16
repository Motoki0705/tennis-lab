"""Frame IO helpers for court-detection visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.tasks.base.visualization.frames import load_rgb_frames


@dataclass(frozen=True)
class CourtFrame:
    """A single visualization input frame."""

    name: str
    rgb: np.ndarray  # (H, W, 3) uint8


@dataclass(frozen=True)
class KpFramePrediction:
    """Per-frame keypoint prediction (data-only, no inference dependency)."""

    keypoints_px: np.ndarray  # (K, 2) in original image pixels
    mean_heatmap: np.ndarray  # (h', w') float in [0, 1]


def load_court_frames(
    source: str | Path,
    *,
    max_frames: int | None = None,
) -> list[CourtFrame]:
    """Load an image source into ordered RGB frames.

    Args:
        source: A single image file, a directory of frames, or a glob pattern.
        max_frames: Optional cap on the number of frames.

    Returns:
        Ordered list of :class:`CourtFrame`.
    """
    return [
        CourtFrame(name=name, rgb=rgb)
        for name, rgb in load_rgb_frames(source, max_frames=max_frames)
    ]
