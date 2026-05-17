"""Types for reusable video streaming pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

TFrame = TypeVar("TFrame")


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata reported by a video backend."""

    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class FramePacket(Generic[TFrame]):
    """One decoded video frame with its source index."""

    index: int
    frame: TFrame
    original_size: tuple[int, int]


@dataclass(frozen=True)
class TemporalWindow(Generic[TFrame]):
    """A fixed-length temporal window over decoded frames."""

    start_index: int
    frame_indices: tuple[int, ...]
    frames: tuple[TFrame, ...]


@dataclass(frozen=True)
class TemporalBatch:
    """Batched temporal windows ready for model inference."""

    windows: tuple[TemporalWindow[torch.Tensor], ...]
    tensor: torch.Tensor
