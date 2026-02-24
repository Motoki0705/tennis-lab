"""Shared typed records for ball_detection data flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Visibility = Literal[0, 1, 2]


@dataclass(frozen=True)
class FrameRecord:
    """Single frame reference inside a clip."""

    frame_index: int
    file_name: str
    frame_path: Path


@dataclass(frozen=True)
class LabelRecord:
    """Per-frame annotation in WASB-compatible semantics."""

    file_name: str
    visibility: Visibility
    x: float
    y: float
    status: int
    score: float


@dataclass(frozen=True)
class DetectionRecord:
    """Detector output on a frame."""

    frame_index: int
    x: float
    y: float
    score: float
    visible: bool


@dataclass(frozen=True)
class EventRecord:
    """Per-frame event score vector."""

    frame_index: int
    shot_prob: float
    bounce_prob: float


@dataclass(frozen=True)
class ConfidenceRecord:
    """Aggregated confidence score for pseudo-label reliability."""

    frame_index: int
    confidence: float
    source: str


@dataclass(frozen=True)
class ClipWindow:
    """Temporal clip boundaries in frame indices."""

    start: int
    end: int


@dataclass(frozen=True)
class ClipLayout:
    """Resolved on-disk clip layout."""

    game_name: str
    clip_name: str
    clip_dir: Path
    label_csv: Path
    frames: tuple[FrameRecord, ...]


@dataclass(frozen=True)
class VideoLayout:
    """Resolved on-disk video layout for pseudo generation."""

    game_name: str
    video_name: str
    video_path: Path


@dataclass(frozen=True)
class PathPolicy:
    """Safety policy for filesystem writes."""

    root_dir: Path
    allow_overwrite: bool = False
