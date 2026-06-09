"""Shared sample and batch contracts for ball detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from torch import Tensor


@dataclass(frozen=True)
class FrameLabel:
    """One normalized ball annotation for a frame."""

    visibility: float
    x: float
    y: float
    instance_id: str = ""
    role: str = "target"
    state: str = "visible"


@dataclass(frozen=True)
class ClipWindow:
    """One fixed-length temporal window consumed by the dataset."""

    clip_dir: Path
    frame_names: tuple[str, ...]
    labels: dict[str, tuple[FrameLabel, ...]]
    original_size: tuple[int, int]
    start_index: int


class BallDetectionSample(TypedDict):
    """One supervised ball detection sample.

    Attributes:
        images: Input RGB frames with shape ``(T, 3, H, W)``.
        heatmaps: Target heatmaps with shape ``(T, Hh, Wh)``.
        coords: Padded ball coordinates in original image pixel space with
            shape ``(T, K, 2)`` and ``(x, y)`` ordering.
        visibility: Padded instance visibility mask with shape ``(T, K)``.
        original_size: Original frame size with shape ``(2,)`` in
            ``(width, height)`` ordering.
        heatmap_size: Heatmap size with shape ``(2,)`` in
            ``(width, height)`` ordering.
    """

    images: Tensor
    heatmaps: Tensor
    coords: Tensor
    visibility: Tensor
    original_size: Tensor
    heatmap_size: Tensor


class BallDetectionBatch(TypedDict):
    """One collated supervised ball detection batch.

    Attributes:
        images: Batched RGB frames with shape ``(B, T, 3, H, W)``.
        heatmaps: Batched target heatmaps with shape ``(B, T, Hh, Wh)``.
        coords: Padded ball coordinates in original image pixel space with
            shape ``(B, T, K, 2)`` and ``(x, y)`` ordering.
        visibility: Padded instance visibility mask with shape ``(B, T, K)``.
        original_size: Original frame sizes with shape ``(B, 2)`` in
            ``(width, height)`` ordering.
        heatmap_size: Heatmap sizes with shape ``(B, 2)`` in
            ``(width, height)`` ordering.
    """

    images: Tensor
    heatmaps: Tensor
    coords: Tensor
    visibility: Tensor
    original_size: Tensor
    heatmap_size: Tensor


__all__ = [
    "BallDetectionBatch",
    "BallDetectionSample",
    "ClipWindow",
    "FrameLabel",
]
