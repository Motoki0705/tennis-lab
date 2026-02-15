"""UV trajectory renderers for event detection visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.event_detection.visualization.rendering.timeline import render_timeline_axes
from src.event_detection.visualization.types import RuntimeConfig, UVEventInputs
from src.utils.schema.court import COURT_SKELETON

DEFAULT_BALL_COLOR: str = "#CCFF00"
GT_SHOT_COLOR: str = "#00FF00"
GT_BOUNCE_COLOR: str = "#FFD700"
PRED_SHOT_COLOR: str = "#FF00FF"
PRED_BOUNCE_COLOR: str = "#00D1FF"


def _draw_court_uv(
    ax: Axes, *, kp: np.ndarray, vis: np.ndarray, show_lines: bool
) -> None:
    ax.set_facecolor("#1a1a1a")

    if show_lines:
        for i, j in COURT_SKELETON:
            if bool(vis[i]) and bool(vis[j]):
                ax.plot(
                    [kp[i, 0], kp[j, 0]],
                    [kp[i, 1], kp[j, 1]],
                    c="lime",
                    linewidth=1.5,
                    alpha=0.8,
                )

    for i in range(int(kp.shape[0])):
        if bool(vis[i]):
            ax.scatter(kp[i, 0], kp[i, 1], c="lime", s=25, marker="s", alpha=0.7)


def _event_sets(
    *,
    inputs: UVEventInputs,
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


def render_uv_trajectory(
    ax: Axes,
    *,
    cfg: RuntimeConfig,
    inputs: UVEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> None:
    """Render UV trajectory with GT and optional predicted events."""
    _draw_court_uv(
        ax, kp=inputs.court_kp, vis=inputs.court_vis, show_lines=cfg.show_court_lines
    )

    uv = inputs.ball_uv
    vis = inputs.ball_vis

    ax.plot(uv[:, 0], uv[:, 1], color="#FF6B6B", alpha=0.35, linewidth=1.0, zorder=1)

    ax.scatter(uv[vis, 0], uv[vis, 1], c=DEFAULT_BALL_COLOR, s=30, alpha=0.8, label="Visible")
    if (~vis).any():
        ax.scatter(uv[~vis, 0], uv[~vis, 1], c="gray", s=14, alpha=0.3, label="Not visible")

    if inputs.shot_indices:
        idx = np.asarray(inputs.shot_indices, dtype=int)
        ax.scatter(
            uv[idx, 0], uv[idx, 1], c=GT_SHOT_COLOR, s=120, marker="*", edgecolors="black", linewidths=1.0, label="GT shot", zorder=5
        )
    if inputs.bounce_indices:
        idx = np.asarray(inputs.bounce_indices, dtype=int)
        ax.scatter(
            uv[idx, 0], uv[idx, 1], c=GT_BOUNCE_COLOR, s=90, marker="o", edgecolors="black", linewidths=1.0, label="GT bounce", zorder=5
        )

    if pred_peaks is not None:
        _, _, pred_shot, pred_bounce = _event_sets(inputs=inputs, pred_peaks=pred_peaks)
        if pred_shot:
            idx = np.asarray(sorted(pred_shot), dtype=int)
            ax.scatter(
                uv[idx, 0], uv[idx, 1], facecolors="none", edgecolors=PRED_SHOT_COLOR, s=110, marker="*", linewidths=1.5, label="Pred shot", zorder=6
            )
        if pred_bounce:
            idx = np.asarray(sorted(pred_bounce), dtype=int)
            ax.scatter(
                uv[idx, 0], uv[idx, 1], facecolors="none", edgecolors=PRED_BOUNCE_COLOR, s=90, marker="o", linewidths=1.5, label="Pred bounce", zorder=6
            )

    if 0 <= cfg.frame < uv.shape[0]:
        ax.scatter(
            [uv[cfg.frame, 0]], [uv[cfg.frame, 1]], c="yellow", s=120, marker="D", edgecolors="black", linewidths=1.0, label=f"Frame {cfg.frame}", zorder=10
        )

    scene_id = inputs.meta.get("scene_id", "Unknown")
    ax.set_title(f"UV Trajectory | scene={scene_id} cam={inputs.camera_idx}")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("U (normalized)")
    ax.set_ylabel("V (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def create_uv_multi_figure(
    *,
    cfg: RuntimeConfig,
    inputs: UVEventInputs,
    probs: np.ndarray | None = None,
    pred_peaks: list[list[int]] | None = None,
    pred_scores: list[list[float]] | None = None,
    event_names: list[str] | None = None,
) -> Figure:
    """Create combined UV trajectory + timeline figure."""
    _, num_events = inputs.targets.shape

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(num_events, 2, width_ratios=[1.0, 1.2])

    ax_traj = fig.add_subplot(gs[:, 0])
    render_uv_trajectory(ax_traj, cfg=cfg, inputs=inputs, pred_peaks=pred_peaks)

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
    fig.suptitle(f"UV trajectory + timeline | scene={scene_id} cam={inputs.camera_idx}")
    plt.tight_layout()
    return fig


def create_uv_animation(
    *,
    cfg: RuntimeConfig,
    inputs: UVEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> FuncAnimation:
    """Create UV animation with event-driven ball coloring."""
    uv = inputs.ball_uv
    num_frames = int(uv.shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_court_uv(ax, kp=inputs.court_kp, vis=inputs.court_vis, show_lines=cfg.show_court_lines)

    (line,) = ax.plot([], [], color="#FF6B6B", alpha=0.5, linewidth=1.2)
    point = ax.scatter([], [], c=DEFAULT_BALL_COLOR, s=120, edgecolors="black", linewidths=2.0, zorder=10)

    gt_shot, gt_bounce, pred_shot, pred_bounce = _event_sets(inputs=inputs, pred_peaks=pred_peaks)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.grid(True, alpha=0.25)

    def update(frame: int) -> tuple[Line2D, PathCollection]:
        line.set_data(uv[: frame + 1, 0], uv[: frame + 1, 1])
        point.set_offsets([[uv[frame, 0], uv[frame, 1]]])
        face, edge = _colors_for_frame(
            frame,
            gt_shot=gt_shot,
            gt_bounce=gt_bounce,
            pred_shot=pred_shot,
            pred_bounce=pred_bounce,
        )
        point.set_facecolor([face])
        point.set_edgecolor([edge])
        ax.set_title(f"UV animation | frame {frame}/{num_frames - 1}")
        return line, point

    return FuncAnimation(fig, update, frames=num_frames, interval=1000.0 / float(cfg.fps), blit=False)
