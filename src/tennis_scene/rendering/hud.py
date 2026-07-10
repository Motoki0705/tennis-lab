"""HUD overlay and top-down minimap for 3D tennis scene animations.

- :class:`HudRenderer` draws frame/time, ball speed, and bounce-count text
  onto the 3D axis with ``Axes3D.text2D`` (so ``ax.clear()`` between frames
  removes it — no artist bookkeeping needed).
- :class:`MinimapRenderer` draws a top-down 2D court inset with player dots,
  the ball with a recent trail, and accumulated bounce marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.utils.rendering.court_renderer import CourtRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray

    from src.tennis_scene.io import SceneResult

MS_TO_KMH: float = 3.6


@dataclass
class HudStyle:
    """Style configuration for the HUD text block.

    Attributes:
        show_frame_info: Show frame index and clock time.
        show_ball_speed: Show ball speed in km/h (requires ``ball_3d``).
        show_bounce_count: Show number of detected bounces so far.
        text_color: HUD text color.
        font_size: HUD font size in points.
    """

    show_frame_info: bool = True
    show_ball_speed: bool = True
    show_bounce_count: bool = True
    text_color: str = "white"
    font_size: float = 11.0


class HudRenderer:
    """Render textual overlays onto a 3D scene axis."""

    def __init__(self, style: HudStyle | None = None) -> None:
        self.style = style or HudStyle()

    def render(
        self,
        ax: Axes3D,
        *,
        frame_idx: int,
        num_frames: int,
        fps: float,
        ball_speed_ms: float | None,
        bounce_count: int | None,
    ) -> None:
        """Draw the HUD text block in the top-left corner of the axis.

        Args:
            ax: Target 3D axis.
            frame_idx: Current frame index.
            num_frames: Total frame count of the scene.
            fps: Scene frame rate (for the clock readout).
            ball_speed_ms: Ball speed in m/s; None hides the line, NaN shows
                a placeholder (ball not tracked on this frame).
            bounce_count: Bounces detected up to this frame; None hides the
                line.
        """
        style = self.style
        lines: list[str] = []
        if style.show_frame_info:
            seconds = frame_idx / fps
            lines.append(f"Frame {frame_idx}/{num_frames}   t={seconds:6.2f}s")
        if style.show_ball_speed and ball_speed_ms is not None:
            if np.isfinite(ball_speed_ms):
                lines.append(f"Ball speed {ball_speed_ms * MS_TO_KMH:5.1f} km/h")
            else:
                lines.append("Ball speed   --  km/h")
        if style.show_bounce_count and bounce_count is not None:
            lines.append(f"Bounces {bounce_count}")
        if not lines:
            return

        ax.text2D(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=style.font_size,
            color=style.text_color,
            family="monospace",
            verticalalignment="top",
            zorder=100,
        )


@dataclass
class MinimapStyle:
    """Style configuration for the top-down minimap inset.

    Attributes:
        ball_color: Ball dot color.
        ball_trail_frames: Number of past frames in the ball trail.
        bounce_color: Bounce cross-mark color.
        player_marker_size: Player dot size in points^2.
        background_alpha: Alpha of the inset background patch.
    """

    ball_color: str = "#CCFF00"
    ball_trail_frames: int = 30
    bounce_color: str = "#FFD700"
    player_marker_size: float = 45.0
    background_alpha: float = 0.85


class MinimapRenderer:
    """Render a top-down 2D court inset for the current frame."""

    def __init__(
        self,
        style: MinimapStyle | None = None,
        court_renderer: CourtRenderer | None = None,
    ) -> None:
        self.style = style or MinimapStyle()
        self.court_renderer = court_renderer or CourtRenderer()

    def render(
        self,
        ax: Axes,
        scene: SceneResult,
        frame_idx: int,
        *,
        player_colors: list[str],
        bounce_frames: NDArray[np.int64] | None = None,
    ) -> None:
        """Draw the minimap for ``frame_idx`` onto a 2D axis.

        Args:
            ax: Target 2D axis (cleared by the caller between frames).
            scene: Scene result with player positions and optional ball track.
            frame_idx: Current frame index.
            player_colors: Per-player colors, aligned with the player axis.
            bounce_frames: Bounce frame indices; marks up to ``frame_idx``.
        """
        style = self.style
        self.court_renderer.render_2d(ax, show_surface=True, set_limits=True)

        num_players = scene.player_position.shape[0]
        if len(player_colors) < num_players:
            raise ValueError(
                f"player_colors has {len(player_colors)} entries for "
                f"{num_players} players"
            )
        for player_idx in range(num_players):
            pos = scene.player_position[player_idx, frame_idx]
            if not np.isfinite(pos[:2]).all():
                continue
            ax.scatter(
                pos[0],
                pos[1],
                c=player_colors[player_idx],
                s=style.player_marker_size,
                zorder=10,
                edgecolors="white",
                linewidths=1.0,
            )

        if scene.ball_3d is not None:
            if bounce_frames is not None:
                past = bounce_frames[bounce_frames <= frame_idx]
                for b in past.tolist():
                    bounce_pos = scene.ball_3d[b]
                    if np.isfinite(bounce_pos[:2]).all():
                        ax.scatter(
                            bounce_pos[0],
                            bounce_pos[1],
                            c=style.bounce_color,
                            marker="x",
                            s=40,
                            linewidths=1.5,
                            zorder=11,
                        )

            trail_start = max(0, frame_idx - style.ball_trail_frames)
            trail = scene.ball_3d[trail_start : frame_idx + 1]
            valid = np.isfinite(trail).all(axis=-1)
            if valid.sum() > 1:
                ax.plot(
                    trail[valid, 0],
                    trail[valid, 1],
                    color=style.ball_color,
                    linewidth=1.2,
                    alpha=0.7,
                    zorder=12,
                )
            ball_pos = scene.ball_3d[frame_idx]
            if np.isfinite(ball_pos[:2]).all():
                ax.scatter(
                    ball_pos[0],
                    ball_pos[1],
                    c=style.ball_color,
                    s=25,
                    zorder=13,
                    edgecolors="black",
                    linewidths=0.5,
                )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.patch.set_alpha(style.background_alpha)
        for spine in ax.spines.values():
            spine.set_color("#888888")
