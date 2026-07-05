"""Image-size arithmetic shared by resize transforms and inference preprocessing.

Consolidates the "resize so the short side matches, then floor both sides to a
multiple of 8" integer math that was duplicated across the court-detection
resize transforms and :func:`preprocess_court_image`.
"""

from __future__ import annotations

__all__ = ["resize_short_side_aligned"]


def resize_short_side_aligned(
    width: int,
    height: int,
    short_side: int,
    *,
    align: int = 8,
) -> tuple[int, int]:
    """Return ``(new_width, new_height)`` with the short side set to ``short_side``.

    The long side is scaled to preserve aspect ratio (``round``), then **both**
    sides are floored to a multiple of ``align``.  This reproduces the exact
    arithmetic previously copy-pasted across resize transforms::

        if height <= width:
            new_h = short_side; new_w = round(width * new_h / height)
        else:
            new_w = short_side; new_h = round(height * new_w / width)
        new_h = (new_h // align) * align
        new_w = (new_w // align) * align

    Args:
        width: Original image width in pixels.
        height: Original image height in pixels.
        short_side: Target length of the shorter side (before alignment).
        align: Both output sides are floored to a multiple of this value.

    Returns:
        Tuple ``(new_width, new_height)``.
    """
    if height <= width:
        new_h = short_side
        new_w = int(round(width * new_h / height))
    else:
        new_w = short_side
        new_h = int(round(height * new_w / width))
    new_h = (new_h // align) * align
    new_w = (new_w // align) * align
    return new_w, new_h
