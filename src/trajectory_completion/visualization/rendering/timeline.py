"""Timeline rendering for trajectory completion visualization."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from src.trajectory_completion.visualization.types import RuntimeConfig, TrajectoryInputs


def render_timeline_panel(
    ax_mask: Axes,
    ax_err: Axes,
    *,
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray | None = None,
) -> None:
    """Render masking and error timeline panels."""
    t_len = int(inputs.ball_uv_gt.shape[0])
    t = np.arange(t_len)

    orig_vis = inputs.ball_gt_visible
    obs = inputs.ball_obs_mask

    newly_masked = orig_vis & (~obs)
    orig_missing = ~orig_vis

    ax_mask.plot(t, orig_vis.astype(np.float32), color="white", alpha=0.4, label="GT visible")
    ax_mask.plot(t, obs.astype(np.float32), color="lime", alpha=0.8, label="Observed (input)")

    if orig_missing.any():
        ax_mask.scatter(
            t[orig_missing],
            np.zeros_like(t[orig_missing]),
            s=14,
            c="gray",
            marker="x",
            alpha=0.7,
            label="Missing in scene",
        )
    if newly_masked.any():
        ax_mask.scatter(
            t[newly_masked],
            np.zeros_like(t[newly_masked]),
            s=16,
            c="#FF4444",
            marker="x",
            alpha=0.9,
            label="Masked by augmentation",
        )

    ax_mask.set_ylim(-0.2, 1.2)
    ax_mask.set_yticks([0.0, 1.0])
    ax_mask.set_title("Visibility / Observation Mask")
    ax_mask.set_xlabel("Frame")
    ax_mask.grid(True, alpha=0.25)
    ax_mask.legend(loc="upper right", fontsize=8)

    err_in = np.linalg.norm(inputs.ball_uv_in - inputs.ball_uv_gt, axis=-1)
    ax_err.scatter(t[obs], err_in[obs], s=10, c="lime", alpha=0.8, label="|Input - GT| (observed)")

    if pred_uv is not None:
        err_pred = np.linalg.norm(pred_uv - inputs.ball_uv_gt, axis=-1)
        ax_err.scatter(t[obs], err_pred[obs], s=10, c="#00D1FF", alpha=0.8, label="|Pred - GT| @ observed")
        ax_err.scatter(t[~obs], err_pred[~obs], s=10, c="#FF00FF", alpha=0.8, label="|Pred - GT| @ masked")

    if cfg.error_threshold > 0:
        ax_err.axhline(
            float(cfg.error_threshold),
            color="yellow",
            alpha=0.5,
            linewidth=1.2,
            linestyle="--",
            label=f"threshold={cfg.error_threshold:g}",
        )

    ax_err.set_title("Per-frame L2 Error (UV units)")
    ax_err.set_xlabel("Frame")
    ax_err.set_ylabel("L2")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="upper right", fontsize=8)
