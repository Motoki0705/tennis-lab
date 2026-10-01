"""Generic bounding-box geometry helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bbox_max_side_ratio(
    box_w: float,
    box_h: float,
    img_w: float,
    img_h: float,
) -> float:
    """Return the larger of ``box_w / img_w`` and ``box_h / img_h``.

    This measures how dominant a box is relative to the frame: the fraction of
    the image side spanned by the matching box side, taken over whichever side
    is more dominant. For already-normalized boxes pass ``img_w = img_h = 1.0``.
    """
    if img_w <= 0.0 or img_h <= 0.0:
        raise ValueError(f"image size must be positive, got ({img_w}, {img_h}).")
    return max(box_w / img_w, box_h / img_h)


def pairwise_iou(first: NDArray[np.floating], second: NDArray[np.floating]) -> NDArray[np.float64]:
    """IoU of every pair of ``(N, 4)`` and ``(M, 4)`` xyxy boxes, ``(N, M)``; 0 where the union is empty."""
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if first.ndim != 2 or first.shape[1] != 4 or second.ndim != 2 or second.shape[1] != 4:
        raise ValueError("Boxes must be (N, 4) xyxy")
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(-1)
    area_first = np.clip(first[:, 2:] - first[:, :2], 0, None).prod(-1)
    area_second = np.clip(second[:, 2:] - second[:, :2], 0, None).prod(-1)
    union = area_first[:, None] + area_second[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


__all__ = ["bbox_max_side_ratio", "pairwise_iou"]
