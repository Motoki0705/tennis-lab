"""Shared sample and batch contracts for ball detection."""

from __future__ import annotations

from dataclasses import dataclass
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


class CandidateReference(TypedDict):
    """Unaugmented validation identity/GT; tensor axes gain B on collation.

    xy is (T,2) in stored pixels, observed is (T,) and requires exactly one
    observed instance. frame_id is a row in the explicitly named namespace.
    source_scale is stored pixels/source pixel, identical on both axes.
    Empty camera means the source has no camera identity.
    """

    xy: Tensor
    observed: Tensor
    frame_id: Tensor
    window_start: Tensor
    source_scale: Tensor
    namespace: str
    camera: str


class BallDetectionSample(TypedDict):
    """One supervised ball detection sample.

    Attributes:
        images: Input RGB frames with shape ``(T, 3, H, W)``.
        heatmaps: Target heatmaps with shape ``(T, Hh, Wh)``.
        coords: Padded ball coordinates in original image pixel space with
            shape ``(T, K, 2)`` and ``(x, y)`` ordering.
        visibility: Padded instance visibility mask with shape ``(T, K)``.
        supervised: Boolean ``(T,)`` mask of the frames whose heatmap target
            is trusted. Loss and metrics ignore the other frames entirely.
        original_size: Original frame size with shape ``(2,)`` in
            ``(width, height)`` ordering.
        heatmap_size: Heatmap size with shape ``(2,)`` in
            ``(width, height)`` ordering.
        window_id: Stable window identifier for persisted predictions.
        source: Name of the data source the window came from.
    """

    images: Tensor
    heatmaps: Tensor
    coords: Tensor
    visibility: Tensor
    supervised: Tensor
    original_size: Tensor
    heatmap_size: Tensor
    window_id: str
    source: str
    candidate_reference: CandidateReference


class BallDetectionBatch(TypedDict):
    """One collated supervised ball detection batch.

    Attributes:
        images: Batched RGB frames with shape ``(B, T, 3, H, W)``.
        heatmaps: Batched target heatmaps with shape ``(B, T, Hh, Wh)``.
        coords: Padded ball coordinates in original image pixel space with
            shape ``(B, T, K, 2)`` and ``(x, y)`` ordering.
        visibility: Padded instance visibility mask with shape ``(B, T, K)``.
        supervised: Boolean ``(B, T)`` supervised-frame mask.
        original_size: Original frame sizes with shape ``(B, 2)`` in
            ``(width, height)`` ordering.
        heatmap_size: Heatmap sizes with shape ``(B, 2)`` in
            ``(width, height)`` ordering.
        window_id: Stable window identifiers for persisted predictions.
        source: Data source name of every window.
    """

    images: Tensor
    heatmaps: Tensor
    coords: Tensor
    visibility: Tensor
    supervised: Tensor
    original_size: Tensor
    heatmap_size: Tensor
    window_id: list[str]
    source: list[str]
    candidate_reference: dict[str, Tensor | list[str]]


__all__ = [
    "BallDetectionBatch",
    "BallDetectionSample",
    "FrameLabel",
]
