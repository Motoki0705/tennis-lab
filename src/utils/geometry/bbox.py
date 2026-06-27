"""Generic bounding-box geometry helpers."""

from __future__ import annotations


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


__all__ = ["bbox_max_side_ratio"]
