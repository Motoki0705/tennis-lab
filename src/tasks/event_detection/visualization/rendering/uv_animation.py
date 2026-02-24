"""UV animation renderer for event detection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from src.tasks.event_detection.visualization.rendering.event_emphasis import (
    build_event_impact,
    mix_color,
)
from src.tasks.event_detection.visualization.types import RuntimeConfig, UVEventInputs
from src.utils.schema.court import COURT_SKELETON

BASE_BALL_RGB = np.asarray([0.80, 1.00, 0.00], dtype=np.float32)
SHOT_RGB = np.asarray([1.00, 0.00, 1.00], dtype=np.float32)
BOUNCE_RGB = np.asarray([0.00, 0.82, 1.00], dtype=np.float32)
TRAIL_COLOR = "#FF6B6B"


def _draw_court_uv(ax: Axes, *, kp: np.ndarray, vis: np.ndarray, show_lines: bool) -> None:
    ax.set_facecolor("#1A1A1A")
    if show_lines:
        for i, j in COURT_SKELETON:
            if bool(vis[i]) and bool(vis[j]):
                ax.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]], c="lime", linewidth=1.5, alpha=0.8)
    for i in range(int(kp.shape[0])):
        if bool(vis[i]):
            ax.scatter(kp[i, 0], kp[i, 1], c="lime", s=25, marker="s", alpha=0.7)


def create_uv_event_animation(
    *,
    cfg: RuntimeConfig,
    inputs: UVEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> FuncAnimation:
    """Animate GT UV trajectory with predicted-event-only color emphasis."""
    uv = inputs.ball_uv
    num_frames = int(uv.shape[0])

    pred_shot = [int(i) for i in pred_peaks[0]] if pred_peaks and len(pred_peaks) > 0 else []
    pred_bounce = [int(i) for i in pred_peaks[1]] if pred_peaks and len(pred_peaks) > 1 else []
    shot_impact = build_event_impact(
        num_frames=num_frames,
        event_indices=pred_shot,
        radius_frames=cfg.event_radius_frames,
        sigma_frames=cfg.event_sigma_frames,
    )
    bounce_impact = build_event_impact(
        num_frames=num_frames,
        event_indices=pred_bounce,
        radius_frames=cfg.event_radius_frames,
        sigma_frames=cfg.event_sigma_frames,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_court_uv(ax, kp=inputs.court_kp, vis=inputs.court_vis, show_lines=cfg.show_court_lines)

    (line,) = ax.plot([], [], color=TRAIL_COLOR, alpha=0.55, linewidth=1.8)
    point = ax.scatter([], [], c=[tuple(BASE_BALL_RGB)], s=120, edgecolors="black", linewidths=2.0, zorder=10)

    ax.plot([], [], color=TRAIL_COLOR, linewidth=1.8, label="Trajectory (GT)")
    ax.scatter([], [], c=[tuple(SHOT_RGB)], s=70, edgecolors="black", linewidths=1.0, label="Pred shot neighborhood")
    ax.scatter([], [], c=[tuple(BOUNCE_RGB)], s=70, edgecolors="black", linewidths=1.0, label="Pred bounce neighborhood")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("U (normalized)")
    ax.set_ylabel("V (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    scene_id = str(inputs.meta.get("scene_id", "Unknown"))

    def update(frame: int) -> tuple[Line2D, PathCollection]:
        line.set_data(uv[: frame + 1, 0], uv[: frame + 1, 1])
        point.set_offsets([[uv[frame, 0], uv[frame, 1]]])

        shot_a = float(shot_impact[frame])
        bounce_a = float(bounce_impact[frame])
        if bounce_a >= shot_a:
            color = mix_color(BASE_BALL_RGB, BOUNCE_RGB, bounce_a)
        else:
            color = mix_color(BASE_BALL_RGB, SHOT_RGB, shot_a)
        point.set_facecolor([color])

        ax.set_title(f"Event detection UV | scene={scene_id} cam={inputs.camera_idx} frame={frame}/{num_frames - 1}")
        return line, point

    return FuncAnimation(fig, update, frames=num_frames, interval=1000.0 / float(cfg.fps), blit=False)

