"""3D trajectory animation renderer for event detection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from src.tasks.event_detection.visualization.rendering.event_emphasis import (
    build_event_impact,
    mix_color,
)
from src.tasks.event_detection.visualization.types import RuntimeConfig, Traj3DEventInputs
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

BASE_BALL_RGB = np.asarray([0.80, 1.00, 0.00], dtype=np.float32)
SHOT_RGB = np.asarray([1.00, 0.00, 1.00], dtype=np.float32)
BOUNCE_RGB = np.asarray([0.00, 0.82, 1.00], dtype=np.float32)
TRAIL_COLOR = "#FF6B6B"


def create_traj3d_event_animation(
    *,
    cfg: RuntimeConfig,
    inputs: Traj3DEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> FuncAnimation:
    """Animate GT 3D trajectory with predicted-event-only color emphasis."""
    pos = inputs.ball_pos_world
    num_frames = int(pos.shape[0])

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

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    court = CourtRenderer()
    court.render_3d(ax, show_net=True)

    (line,) = ax.plot([], [], [], color=TRAIL_COLOR, alpha=0.55, linewidth=2.0)
    point = ax.scatter([], [], [], c=[tuple(BASE_BALL_RGB)], s=120, edgecolors="black", linewidths=1.2)

    ax.plot([], [], [], color=TRAIL_COLOR, linewidth=2.0, label="Trajectory (GT)")
    ax.scatter([], [], [], c=[tuple(SHOT_RGB)], s=70, edgecolors="black", linewidths=1.0, label="Pred shot neighborhood")
    ax.scatter([], [], [], c=[tuple(BOUNCE_RGB)], s=70, edgecolors="black", linewidths=1.0, label="Pred bounce neighborhood")
    ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(-HALF_DOUBLES_WIDTH - 2.0, HALF_DOUBLES_WIDTH + 2.0)
    ax.set_ylim(-HALF_LENGTH - 2.0, HALF_LENGTH + 2.0)
    z_max = max(3.0, float(np.max(pos[:, 2])) + 0.5) if num_frames > 0 else 3.0
    ax.set_zlim(0.0, z_max)

    scene_id = str(inputs.meta.get("scene_id", "Unknown"))

    def update(frame: int) -> tuple[Line2D]:
        line.set_data(pos[: frame + 1, 0], pos[: frame + 1, 1])
        line.set_3d_properties(pos[: frame + 1, 2])
        point._offsets3d = ([pos[frame, 0]], [pos[frame, 1]], [pos[frame, 2]])

        shot_a = float(shot_impact[frame])
        bounce_a = float(bounce_impact[frame])
        if bounce_a >= shot_a:
            color = mix_color(BASE_BALL_RGB, BOUNCE_RGB, bounce_a)
        else:
            color = mix_color(BASE_BALL_RGB, SHOT_RGB, shot_a)
        point.set_facecolor([color])

        ax.set_title(f"Event detection 3D | scene={scene_id} frame={frame}/{num_frames - 1}")
        return (line,)

    return FuncAnimation(fig, update, frames=num_frames, interval=1000.0 / float(cfg.fps), blit=False)

