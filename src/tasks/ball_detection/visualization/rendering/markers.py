"""Subtle prediction markers for textless ball overlays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

OverlayMarker = Literal["point", "ring", "trail"]


@dataclass(frozen=True)
class RingStyle:
    """Ring dimensions at width 960, with a softly blended outer edge."""

    radius: int
    thickness: int
    center_radius: int
    glow_radius: int
    glow_opacity: float


@dataclass(frozen=True)
class TrailStyle:
    """Dot dimensions at width 960; history length counts source frames."""

    radius: int
    length: int
    thickness: int
    opacity: float


def draw_ring(
    panel: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], style: RingStyle,
) -> None:
    """Draw a thin ring, center dot and faint Gaussian glow in-place."""
    height, width = panel.shape[:2]
    scale = width / 960
    radius = max(1, round(style.radius * scale))
    thickness = max(1, round(style.thickness * scale))
    sigma = max(1, round(style.glow_radius * scale))
    margin = radius + 3 * sigma + thickness
    x, y = center
    left, right = max(0, x - margin), min(width, x + margin + 1)
    top, bottom = max(0, y - margin), min(height, y + margin + 1)
    if right > left and bottom > top:
        roi = panel[top:bottom, left:right]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x - left, y - top), radius, 255, thickness, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        alpha = mask[..., None].astype(np.float32) * (style.glow_opacity / 255)
        roi[:] = np.rint(roi * (1 - alpha) + np.asarray(color) * alpha).astype(np.uint8)
    cv2.circle(panel, center, radius, color, thickness, cv2.LINE_AA)
    cv2.circle(panel, center, max(1, round(style.center_radius * scale)), color, -1, cv2.LINE_AA)


def draw_trail(
    panel: np.ndarray,
    history: Sequence[tuple[int, int]],
    color: tuple[int, int, int],
    style: TrailStyle,
) -> None:
    """Draw fading adjacent predictions, with an opaque dot at the newest one."""
    scale = panel.shape[1] / 960
    for index in range(1, len(history)):
        opacity = style.opacity * index / (len(history) - 1)
        layer = panel.copy()
        cv2.line(layer, history[index - 1], history[index], color,
                 max(1, round(style.thickness * scale)), cv2.LINE_AA)
        cv2.addWeighted(layer, opacity, panel, 1 - opacity, 0, dst=panel)
    cv2.circle(panel, history[-1], max(1, round(style.radius * scale)), color, -1, cv2.LINE_AA)
