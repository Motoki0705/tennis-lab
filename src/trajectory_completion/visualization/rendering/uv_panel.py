"""UV panel rendering for trajectory completion visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.trajectory_completion.visualization.rendering.timeline import render_timeline_panel
from src.trajectory_completion.visualization.types import RuntimeConfig, TrajectoryInputs
from src.utils.schema.court import COURT_SKELETON


def _draw_court_uv(
    ax: Axes,
    *,
    court_kp: np.ndarray,
    court_vis: np.ndarray,
    show_lines: bool,
) -> None:
    ax.set_facecolor("#1a1a1a")

    if show_lines:
        for i, j in COURT_SKELETON:
            if bool(court_vis[i]) and bool(court_vis[j]):
                ax.plot(
                    [court_kp[i, 0], court_kp[j, 0]],
                    [court_kp[i, 1], court_kp[j, 1]],
                    c="lime",
                    linewidth=1.5,
                    alpha=0.8,
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


def render_uv_panel(
    ax: Axes,
    *,
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray | None = None,
    completed_uv: np.ndarray | None = None,
) -> None:
    """Render UV comparison panel."""
    _draw_court_uv(
        ax,
        court_kp=inputs.court_kp,
        court_vis=inputs.court_vis,
        show_lines=cfg.show_court_lines,
    )

    gt = inputs.ball_uv_gt
    uv_in = inputs.ball_uv_in
    orig_vis = inputs.ball_gt_visible
    obs = inputs.ball_obs_mask

    orig_missing = ~orig_vis
    newly_masked = orig_vis & (~obs)

    ax.plot(gt[:, 0], gt[:, 1], color="white", alpha=0.25, linewidth=1.0, label="GT")

    if orig_missing.any():
        ax.scatter(
            gt[orig_missing, 0],
            gt[orig_missing, 1],
            c="gray",
            s=18,
            marker="x",
            alpha=0.5,
            label="GT (missing in scene)",
            zorder=4,
        )

    if newly_masked.any():
        ax.scatter(
            gt[newly_masked, 0],
            gt[newly_masked, 1],
            c="#FF4444",
            s=22,
            marker="x",
            alpha=0.9,
            label="GT (masked by augmentation)",
            zorder=5,
        )

    if obs.any():
        ax.scatter(
            uv_in[obs, 0],
            uv_in[obs, 1],
            c="lime",
            s=28,
            marker="o",
            alpha=0.8,
            label="Input (observed)",
            zorder=6,
        )

    if obs.any() and cfg.connector_stride > 0:
        idx = np.where(obs)[0][:: cfg.connector_stride]
        for i in idx:
            ax.plot(
                [gt[i, 0], uv_in[i, 0]],
                [gt[i, 1], uv_in[i, 1]],
                color="orange",
                alpha=0.15,
                linewidth=0.6,
                zorder=3,
            )

    if pred_uv is not None:
        ax.scatter(
            pred_uv[obs, 0],
            pred_uv[obs, 1],
            facecolors="none",
            edgecolors="#00D1FF",
            s=60,
            marker="o",
            linewidths=1.8,
            alpha=0.9,
            label="Pred @ observed",
            zorder=7,
        )
        ax.scatter(
            pred_uv[~obs, 0],
            pred_uv[~obs, 1],
            c="#FF00FF",
            s=55,
            marker="^",
            alpha=0.8,
            label="Pred @ masked",
            zorder=7,
        )

    if completed_uv is not None:
        ax.plot(
            completed_uv[:, 0],
            completed_uv[:, 1],
            color="#00D1FF",
            alpha=0.7,
            linewidth=1.6,
            label="Completed (merge_observed)",
            zorder=6,
        )

    if 0 <= cfg.frame < gt.shape[0]:
        ax.scatter(
            [gt[cfg.frame, 0]],
            [gt[cfg.frame, 1]],
            c="yellow",
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
            label=f"Frame {cfg.frame}",
        )

    scene_id = inputs.meta.get("scene_id", "Unknown")
    ax.set_title(f"UV View | scene={scene_id} cam={inputs.camera_idx}")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("U (normalized)")
    ax.set_ylabel("V (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def create_multi_figure(
    *,
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray | None = None,
    completed_uv: np.ndarray | None = None,
) -> Figure:
    """Create multi-panel figure (UV + mask + error)."""
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0])

    ax_uv = fig.add_subplot(gs[:, 0])
    ax_mask = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])

    render_uv_panel(
        ax_uv,
        cfg=cfg,
        inputs=inputs,
        pred_uv=pred_uv,
        completed_uv=completed_uv,
    )
    render_timeline_panel(ax_mask, ax_err, cfg=cfg, inputs=inputs, pred_uv=pred_uv)

    plt.tight_layout()
    return fig
