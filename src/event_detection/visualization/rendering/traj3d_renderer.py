"""3D trajectory renderers for event detection visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.event_detection.visualization.rendering.timeline import render_timeline_axes
from src.event_detection.visualization.types import RuntimeConfig, Traj3DEventInputs
from src.utils.rendering.court_renderer import CourtRenderer
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

DEFAULT_BALL_COLOR: str = "#CCFF00"
GT_SHOT_COLOR: str = "#00FF00"
GT_BOUNCE_COLOR: str = "#FFD700"
PRED_SHOT_COLOR: str = "#FF00FF"
PRED_BOUNCE_COLOR: str = "#00D1FF"


def _event_sets(
    *,
    inputs: Traj3DEventInputs,
    pred_peaks: list[list[int]] | None,
) -> tuple[set[int], set[int], set[int], set[int]]:
    gt_shot = set(inputs.shot_indices)
    gt_bounce = set(inputs.bounce_indices)

    pred_shot: set[int] = set()
    pred_bounce: set[int] = set()
    if pred_peaks is not None:
        if len(pred_peaks) > 0:
            pred_shot = set(int(i) for i in pred_peaks[0])
        if len(pred_peaks) > 1:
            pred_bounce = set(int(i) for i in pred_peaks[1])

    return gt_shot, gt_bounce, pred_shot, pred_bounce


def _colors_for_frame(
    frame: int,
    *,
    gt_shot: set[int],
    gt_bounce: set[int],
    pred_shot: set[int],
    pred_bounce: set[int],
) -> tuple[str, str]:
    gt_event = "bounce" if frame in gt_bounce else "shot" if frame in gt_shot else None
    pred_event = (
        "bounce" if frame in pred_bounce else "shot" if frame in pred_shot else None
    )

    face = DEFAULT_BALL_COLOR
    edge = "black"

    if gt_event == "bounce":
        face = GT_BOUNCE_COLOR
    elif gt_event == "shot":
        face = GT_SHOT_COLOR
    elif pred_event == "bounce":
        face = PRED_BOUNCE_COLOR
    elif pred_event == "shot":
        face = PRED_SHOT_COLOR

    if gt_event is not None and pred_event is not None:
        edge = PRED_BOUNCE_COLOR if pred_event == "bounce" else PRED_SHOT_COLOR

    return face, edge


def render_topdown_trajectory(
    ax: Axes,
    *,
    inputs: Traj3DEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> None:
    """Render top-down 2D trajectory (XY) with event markers."""
    court = CourtRenderer()
    court.render_2d(ax, show_fence=True, set_limits=False)

    pos = inputs.ball_pos_world
    ax.plot(pos[:, 0], pos[:, 1], color="#FF6B6B", alpha=0.5, linewidth=1.2, label="Trajectory")

    if inputs.shot_indices:
        idx = np.asarray(inputs.shot_indices, dtype=int)
        ax.scatter(pos[idx, 0], pos[idx, 1], c=GT_SHOT_COLOR, s=120, marker="*", edgecolors="black", linewidths=1.0, label="GT shot", zorder=6)
    if inputs.bounce_indices:
        idx = np.asarray(inputs.bounce_indices, dtype=int)
        ax.scatter(pos[idx, 0], pos[idx, 1], c=GT_BOUNCE_COLOR, s=90, marker="o", edgecolors="black", linewidths=1.0, label="GT bounce", zorder=6)

    if pred_peaks is not None:
        _, _, pred_shot, pred_bounce = _event_sets(inputs=inputs, pred_peaks=pred_peaks)
        if pred_shot:
            idx = np.asarray(sorted(pred_shot), dtype=int)
            ax.scatter(pos[idx, 0], pos[idx, 1], facecolors="none", edgecolors=PRED_SHOT_COLOR, s=110, marker="*", linewidths=1.5, label="Pred shot", zorder=7)
        if pred_bounce:
            idx = np.asarray(sorted(pred_bounce), dtype=int)
            ax.scatter(pos[idx, 0], pos[idx, 1], facecolors="none", edgecolors=PRED_BOUNCE_COLOR, s=90, marker="o", linewidths=1.5, label="Pred bounce", zorder=7)

    ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
    ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    scene_id = inputs.meta.get("scene_id", "Unknown")
    ax.set_title(f"3D top-down | scene={scene_id}")
    ax.legend(loc="upper right", fontsize=8)


def create_traj3d_multi_figure(
    *,
    cfg: RuntimeConfig,
    inputs: Traj3DEventInputs,
    probs: np.ndarray | None = None,
    pred_peaks: list[list[int]] | None = None,
    pred_scores: list[list[float]] | None = None,
    event_names: list[str] | None = None,
) -> Figure:
    """Create combined top-down trajectory + timeline figure."""
    _, num_events = inputs.targets.shape

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(num_events, 2, width_ratios=[1.0, 1.2])

    ax_traj = fig.add_subplot(gs[:, 0])
    render_topdown_trajectory(ax_traj, inputs=inputs, pred_peaks=pred_peaks)

    axes: list[Axes] = []
    for event_idx in range(num_events):
        if event_idx == 0:
            ax = fig.add_subplot(gs[event_idx, 1])
        else:
            ax = fig.add_subplot(gs[event_idx, 1], sharex=axes[0])
        axes.append(ax)

    render_timeline_axes(
        axes,
        threshold=cfg.threshold,
        targets=inputs.targets,
        shot_indices=inputs.shot_indices,
        bounce_indices=inputs.bounce_indices,
        probs=probs,
        pred_peaks=pred_peaks,
        pred_scores=pred_scores,
        event_names=event_names,
    )

    scene_id = inputs.meta.get("scene_id", "Unknown")
    fig.suptitle(f"3D top-down + timeline | scene={scene_id}")
    plt.tight_layout()
    return fig


def create_traj3d_animation(
    *,
    cfg: RuntimeConfig,
    inputs: Traj3DEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> FuncAnimation:
    """Create top-down animation with event-driven ball coloring."""
    pos = inputs.ball_pos_world
    num_frames = int(pos.shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    court = CourtRenderer()
    court.render_2d(ax, show_fence=True, set_limits=False)

    (line,) = ax.plot([], [], color="#FF6B6B", alpha=0.5, linewidth=1.2)
    point = ax.scatter([], [], c=DEFAULT_BALL_COLOR, s=120, edgecolors="black", linewidths=2.0, zorder=10)

    gt_shot, gt_bounce, pred_shot, pred_bounce = _event_sets(inputs=inputs, pred_peaks=pred_peaks)

    ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
    ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    def update(frame: int) -> tuple[Line2D, PathCollection]:
        line.set_data(pos[: frame + 1, 0], pos[: frame + 1, 1])
        point.set_offsets([[pos[frame, 0], pos[frame, 1]]])
        face, edge = _colors_for_frame(
            frame,
            gt_shot=gt_shot,
            gt_bounce=gt_bounce,
            pred_shot=pred_shot,
            pred_bounce=pred_bounce,
        )
        point.set_facecolor([face])
        point.set_edgecolor([edge])
        ax.set_title(f"3D top-down animation | frame {frame}/{num_frames - 1}")
        return line, point

    return FuncAnimation(fig, update, frames=num_frames, interval=1000.0 / float(cfg.fps), blit=False)
