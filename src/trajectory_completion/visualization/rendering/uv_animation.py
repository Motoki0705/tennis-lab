"""UV animation renderer for trajectory completion visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from src.trajectory_completion.visualization.types import RuntimeConfig, TrajectoryInputs
from src.utils.schema.court import COURT_SKELETON

GT_LINE_COLOR = "#F5F5F5"
OBSERVED_POINT_COLOR = "#00D1FF"
COMPLETED_POINT_COLOR = "#FF00FF"


def _draw_court_uv(
    ax: Axes,
    *,
    court_kp: np.ndarray,
    court_vis: np.ndarray,
    show_lines: bool,
) -> None:
    ax.set_facecolor("#1A1A1A")

    if show_lines:
        for i, j in COURT_SKELETON:
            if bool(court_vis[i]) and bool(court_vis[j]):
                ax.plot(
                    [court_kp[i, 0], court_kp[j, 0]],
                    [court_kp[i, 1], court_kp[j, 1]],
                    c="lime",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=1,
                )

    for i in range(int(court_kp.shape[0])):
        if bool(court_vis[i]):
            ax.scatter(
                court_kp[i, 0],
                court_kp[i, 1],
                c="lime",
                s=25,
                marker="s",
                alpha=0.7,
                zorder=2,
            )


def create_uv_completion_animation(
    *,
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray | None,
) -> FuncAnimation:
    """Create animation with GT line and prediction point color-coded by completion."""
    gt = inputs.ball_uv_gt
    obs_mask = inputs.ball_obs_mask
    if pred_uv is not None:
        point_xy = pred_uv
    else:
        # In visualize mode, missing points in ball_uv_in are often [0, 0].
        # Use GT coordinates for unobserved frames to avoid plotting at the corner.
        point_xy = np.where(obs_mask[:, None], inputs.ball_uv_in, gt)
    num_frames = int(gt.shape[0])
    fps = float(cfg.fps) if cfg.fps is not None else 30.0

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_court_uv(
        ax,
        court_kp=inputs.court_kp,
        court_vis=inputs.court_vis,
        show_lines=cfg.show_court_lines,
    )

    (gt_line,) = ax.plot([], [], color=GT_LINE_COLOR, linewidth=2.0, alpha=0.9, zorder=4)
    pred_point = ax.scatter(
        [],
        [],
        c=[OBSERVED_POINT_COLOR],
        s=130,
        edgecolors="black",
        linewidths=1.8,
        zorder=8,
    )

    ax.plot([], [], color=GT_LINE_COLOR, linewidth=2.0, label="GT trajectory")
    ax.scatter([], [], c=OBSERVED_POINT_COLOR, s=80, edgecolors="black", linewidths=1.0, label="Pred @ observed")
    ax.scatter([], [], c=COMPLETED_POINT_COLOR, s=80, edgecolors="black", linewidths=1.0, label="Pred @ completed")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("U (normalized)")
    ax.set_ylabel("V (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    scene_id = str(inputs.meta.get("scene_id", "Unknown"))

    def update(frame: int) -> tuple[Line2D, PathCollection]:
        gt_line.set_data(gt[: frame + 1, 0], gt[: frame + 1, 1])
        pred_point.set_offsets([[point_xy[frame, 0], point_xy[frame, 1]]])
        point_color = OBSERVED_POINT_COLOR if bool(obs_mask[frame]) else COMPLETED_POINT_COLOR
        pred_point.set_facecolor([point_color])
        ax.set_title(f"Trajectory completion | scene={scene_id} cam={inputs.camera_idx} frame={frame}/{num_frames - 1}")
        return gt_line, pred_point

    return FuncAnimation(fig, update, frames=num_frames, interval=1000.0 / fps, blit=False)
