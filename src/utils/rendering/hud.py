"""Generic HUD text overlay for 3D scene rendering.

:func:`render_hud_text` draws caller-provided text lines onto a 3D axis with
``Axes3D.text2D`` (so ``ax.clear()`` between frames removes it — no artist
bookkeeping needed). It deliberately knows nothing about scenes, balls, or
bounces; tasks select and format their own lines, optionally with the
formatting helpers here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.utils.rendering.layers import SceneLayer

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

MS_TO_KMH: float = 3.6


@dataclass(frozen=True)
class HudStyle:
    """Style configuration for the HUD text block.

    Attributes:
        text_color: HUD text color.
        font_size: HUD font size in points.
        x: Horizontal anchor in axes coordinates.
        y: Vertical anchor in axes coordinates (text grows downward).
    """

    text_color: str = "white"
    font_size: float = 11.0
    x: float = 0.02
    y: float = 0.98


def format_frame_clock(frame_idx: int, num_frames: int, fps: float) -> str:
    """Frame counter with wall-clock time, e.g. ``Frame 45/90   t=  1.50s``."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    seconds = frame_idx / fps
    return f"Frame {frame_idx}/{num_frames}   t={seconds:6.2f}s"


def format_speed_kmh(speed_ms: float) -> str:
    """Speed in km/h, e.g. `` 36.0 km/h``; a placeholder when non-finite."""
    if np.isfinite(speed_ms):
        return f"{speed_ms * MS_TO_KMH:5.1f} km/h"
    return "  --  km/h"


def render_hud_text(
    ax: Axes3D,
    lines: Sequence[str],
    style: HudStyle | None = None,
) -> None:
    """Draw ``lines`` as a monospace text block onto a 3D axis.

    Args:
        ax: Target 3D axis.
        lines: Text lines, top to bottom. Nothing is drawn when empty.
        style: Text style; defaults to :class:`HudStyle`.
    """
    if not lines:
        return
    style = style or HudStyle()
    ax.text2D(
        style.x,
        style.y,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=style.font_size,
        color=style.text_color,
        family="monospace",
        verticalalignment="top",
        zorder=SceneLayer.OVERLAY,
    )
