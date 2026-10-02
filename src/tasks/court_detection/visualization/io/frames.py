"""Frame IO helpers for court-detection visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.dataset.court.sample_store import (
    STORAGE_SCHEMA,
    open_court_store,
)
from src.tasks.base.visualization.frames import load_rgb_frames
from src.utils.data.image_record_store import ImageRecordStore


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
    path = Path(source)
    if path.name == "dataset.json" and path.is_file():
        descriptor = json.loads(path.read_text())
        if descriptor.get("schema") == STORAGE_SCHEMA:
            store = open_court_store(path.parent)
        elif descriptor.get("schema") == "tennis_court_detector_store_v1":
            store = ImageRecordStore(path.parent, descriptor["storage"])
        else:
            raise ValueError("Unsupported Court visualization store schema.")
        count = len(store) if max_frames is None else min(len(store), max_frames)
        return [CourtFrame(name=str(store.record(row).get("sample_id", store.record(row).get("id"))), rgb=store.rgb(row)) for row in range(count)]
    return [
        CourtFrame(name=name, rgb=rgb)
        for name, rgb in load_rgb_frames(source, max_frames=max_frames)
    ]
