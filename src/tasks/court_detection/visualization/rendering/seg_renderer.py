"""Segmentation visualization renderer for court detection."""

from __future__ import annotations

import numpy as np

from src.tasks.court_detection.visualization.io.frames import CourtFrame
from src.tasks.court_detection.visualization.rendering.common import (
    CourtRenderStyle,
    colorize_seg_mask,
    compose_two_panel,
    resize_for_display,
)


def render_seg_frames(
    *,
    frames: list[CourtFrame],
    masks: list[np.ndarray],
    style: CourtRenderStyle,
    clip_label: str,
) -> list[np.ndarray]:
    """Render ``[RGB | colorized segmentation map]`` frames."""
    total = len(frames)
    rendered: list[np.ndarray] = []
    for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        left = resize_for_display(frame.rgb, style.display_width)
        right = colorize_seg_mask(mask)
        header = f"{clip_label} | frame {index + 1}/{total} | {frame.name}"
        rendered.append(
            compose_two_panel(
                left_rgb=left,
                right_rgb=right,
                left_label="RGB",
                right_label="seg map",
                header_text=header,
                style=style,
            )
        )
    return rendered
