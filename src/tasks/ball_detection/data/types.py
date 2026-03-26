"""Shared sample and batch contracts for ball detection."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class BallDetectionSample(TypedDict):
    """One supervised ball detection sample.

    Attributes:
        images: Input RGB frames with shape ``(T, 3, H, W)``.
        heatmaps: Target heatmaps with shape ``(T, Hh, Wh)``.
        coords: Ball coordinates in original image pixel space with shape
            ``(T, 2)`` and ``(x, y)`` ordering.
        visibility: Frame visibility flags with shape ``(T,)``.
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
        coords: Ball coordinates in original image pixel space with shape
            ``(B, T, 2)`` and ``(x, y)`` ordering.
        visibility: Frame visibility flags with shape ``(B, T)``.
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


__all__ = ["BallDetectionBatch", "BallDetectionSample"]
