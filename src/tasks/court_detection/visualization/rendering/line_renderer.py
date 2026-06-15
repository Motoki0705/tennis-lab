"""Line visualization renderer for court detection."""

from __future__ import annotations

import numpy as np

from src.tasks.court_detection.visualization.io.frames import CourtFrame
from src.tasks.court_detection.visualization.rendering.common import (
    CourtRenderStyle,
    compose_two_panel,
    line_prob_to_map,
    resize_for_display,
)


def render_line_frames(
    *,
    frames: list[CourtFrame],
    probs: list[np.ndarray],
    style: CourtRenderStyle,
    clip_label: str,
) -> list[np.ndarray]:
    """Render ``[RGB | line probability map]`` frames."""
    total = len(frames)
    rendered: list[np.ndarray] = []
    for index, (frame, prob) in enumerate(zip(frames, probs, strict=True)):
        left = resize_for_display(frame.rgb, style.display_width)
        right = line_prob_to_map(prob)
        header = f"{clip_label} | frame {index + 1}/{total} | {frame.name}"
        rendered.append(
            compose_two_panel(
                left_rgb=left,
                right_rgb=right,
                left_label="RGB",
                right_label="line map",
                header_text=header,
                style=style,
            )
        )
    return rendered
