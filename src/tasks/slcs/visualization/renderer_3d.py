"""3D / top-down qualitative rendering of SLCS clip predictions.

Renders prediction-vs-pseudo-label comparisons on the standard court:
players as position markers with yaw arrows, the ball as a trajectory trail.
Frames are drawn with matplotlib (Agg), converted to RGB arrays and written
as H.264 video via :func:`src.utils.video.writer.save_video_rgb`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.video.writer import save_video_rgb

_PLAYER_COLORS = ("tab:blue", "tab:orange")
_ARROW_LEN_M = 1.2
_TRAIL_FRAMES = 24


@dataclass(frozen=True)
class SceneRenderInputs:
    """Full-timeline arrays (meters / radians) for one clip camera.

    Ground-truth arrays are the pseudo-labels; ``None`` disables the GT layer.
    Validity masks control which frames draw markers.
    """

    player_position_m: NDArray[np.float32]  # (P, T, 3)
    player_yaw_rad: NDArray[np.float32]  # (P, T)
    ball_position_m: NDArray[np.float32]  # (T, 3)
    gt_player_position_m: NDArray[np.float32] | None = None  # (P, T, 3)
    gt_player_yaw_rad: NDArray[np.float32] | None = None  # (P, T)
    gt_ball_position_m: NDArray[np.float32] | None = None  # (T, 3)
    gt_player_valid: NDArray[np.bool_] | None = None  # (P, T)
    gt_ball_valid: NDArray[np.bool_] | None = None  # (T,)

    def __post_init__(self) -> None:
        if self.player_position_m.ndim != 3 or self.player_position_m.shape[2] != 3:
            raise ValueError(
                f"player_position_m must be (P, T, 3), got {self.player_position_m.shape}."
            )
        num_players, num_frames = self.player_position_m.shape[:2]
        if self.player_yaw_rad.shape != (num_players, num_frames):
            raise ValueError(
                f"player_yaw_rad must be (P, T)={(num_players, num_frames)}, "
                f"got {self.player_yaw_rad.shape}."
            )
        if self.ball_position_m.shape != (num_frames, 3):
            raise ValueError(
                f"ball_position_m must be (T, 3)=({num_frames}, 3), "
                f"got {self.ball_position_m.shape}."
            )

    @property
    def num_frames(self) -> int:
        return int(self.player_position_m.shape[1])

    @property
    def num_players(self) -> int:
        return int(self.player_position_m.shape[0])


class SLCSSceneRenderer:
    """Render an SLCS scene timeline to video (3D view + top-down view)."""

    def __init__(self, *, figsize: tuple[float, float] = (12.0, 6.0), dpi: int = 100) -> None:
        self.figsize = figsize
        self.dpi = dpi
        self._court = CourtRenderer()

    def render_video(
        self,
        inputs: SceneRenderInputs,
        output_path: str | Path,
        *,
        fps: float = 30.0,
        frame_step: int = 1,
    ) -> Path:
        """Render the timeline and write an H.264 video; returns the path."""
        if frame_step <= 0:
            raise ValueError(f"frame_step must be positive, got {frame_step}.")
        frames = [
            self.render_frame(inputs, t)
            for t in range(0, inputs.num_frames, frame_step)
        ]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_video_rgb(np.stack(frames), output_path, fps=float(fps) / frame_step)
        return output_path

    def render_frame(self, inputs: SceneRenderInputs, t: int) -> NDArray[np.uint8]:
        """Render one timeline frame to an RGB uint8 array."""
        if not 0 <= t < inputs.num_frames:
            raise ValueError(f"frame {t} out of range [0, {inputs.num_frames}).")
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)
        self._draw_3d(ax3d, inputs, t)
        self._draw_topdown(ax2d, inputs, t)
        fig.suptitle(f"SLCS frame {t}")
        fig.tight_layout()
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        return np.ascontiguousarray(buffer)

    # ------------------------------------------------------------------

    def _draw_3d(self, ax: object, inputs: SceneRenderInputs, t: int) -> None:
        self._court.render_3d(ax)
        trail = slice(max(0, t - _TRAIL_FRAMES), t + 1)
        ball = inputs.ball_position_m
        ax.plot(  # type: ignore[attr-defined]
            ball[trail, 0], ball[trail, 1], ball[trail, 2],
            color="tab:green", linewidth=2, label="ball (pred)",
        )
        ax.scatter(  # type: ignore[attr-defined]
            [ball[t, 0]], [ball[t, 1]], [ball[t, 2]], color="tab:green", s=40
        )
        if inputs.gt_ball_position_m is not None:
            gt_ball = inputs.gt_ball_position_m
            ax.plot(  # type: ignore[attr-defined]
                gt_ball[trail, 0], gt_ball[trail, 1], gt_ball[trail, 2],
                color="tab:green", linewidth=1, linestyle="--", alpha=0.6,
                label="ball (label)",
            )
        for p in range(inputs.num_players):
            pos = inputs.player_position_m[p, t]
            ax.scatter(  # type: ignore[attr-defined]
                [pos[0]], [pos[1]], [pos[2]], color=_PLAYER_COLORS[p % 2], s=60,
                label=f"player {p} (pred)",
            )
            if inputs.gt_player_position_m is not None:
                gt = inputs.gt_player_position_m[p, t]
                ax.scatter(  # type: ignore[attr-defined]
                    [gt[0]], [gt[1]], [gt[2]], color=_PLAYER_COLORS[p % 2], s=40,
                    marker="x", alpha=0.7,
                )
        ax.legend(loc="upper left", fontsize=7)  # type: ignore[attr-defined]

    def _draw_topdown(self, ax: plt.Axes, inputs: SceneRenderInputs, t: int) -> None:
        self._court.render_2d(ax, show_fence=True)
        trail = slice(max(0, t - _TRAIL_FRAMES), t + 1)
        ball = inputs.ball_position_m
        ax.plot(ball[trail, 0], ball[trail, 1], color="tab:green", linewidth=2)
        ax.scatter([ball[t, 0]], [ball[t, 1]], color="tab:green", s=40)
        for p in range(inputs.num_players):
            color = _PLAYER_COLORS[p % 2]
            pos = inputs.player_position_m[p, t]
            yaw = float(inputs.player_yaw_rad[p, t])
            ax.scatter([pos[0]], [pos[1]], color=color, s=60)
            ax.arrow(
                float(pos[0]), float(pos[1]),
                _ARROW_LEN_M * np.cos(yaw), _ARROW_LEN_M * np.sin(yaw),
                color=color, width=0.05, head_width=0.35, length_includes_head=True,
            )
            if inputs.gt_player_position_m is not None:
                valid = (
                    inputs.gt_player_valid is None or bool(inputs.gt_player_valid[p, t])
                )
                if valid:
                    gt = inputs.gt_player_position_m[p, t]
                    ax.scatter([gt[0]], [gt[1]], color=color, s=40, marker="x", alpha=0.7)
                    if inputs.gt_player_yaw_rad is not None:
                        gt_yaw = float(inputs.gt_player_yaw_rad[p, t])
                        ax.arrow(
                            float(gt[0]), float(gt[1]),
                            _ARROW_LEN_M * np.cos(gt_yaw), _ARROW_LEN_M * np.sin(gt_yaw),
                            color=color, width=0.02, head_width=0.2, alpha=0.5,
                            length_includes_head=True, linestyle="--",
                        )
        ax.set_title("top-down (pred=solid, label=x/dashed)", fontsize=8)


__all__ = ["SceneRenderInputs", "SLCSSceneRenderer"]
