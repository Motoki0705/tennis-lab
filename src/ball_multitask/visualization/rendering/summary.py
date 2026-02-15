"""Summary rendering for ball multitask outputs."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.ball_multitask.visualization.types import SceneInputs


def create_summary_figure(
    *,
    inputs: SceneInputs,
    outputs: dict[str, Any],
) -> plt.Figure:
    """Create a 3-panel summary figure for prediction outputs."""
    t = np.arange(inputs.seq_len)
    uv_gt = inputs.ball_uv[: inputs.seq_len]
    uv_pred = np.asarray(outputs["uv_completed"], dtype=np.float32)
    pos3d = np.asarray(outputs["position_3d"], dtype=np.float32)
    probs = np.asarray(outputs["event_probs"], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax_uv = axes[0]
    ax_uv.plot(t, uv_gt[:, 0], label="gt_u", color="tab:blue")
    ax_uv.plot(t, uv_gt[:, 1], label="gt_v", color="tab:cyan")
    ax_uv.plot(t, uv_pred[:, 0], label="pred_u", color="tab:orange", linestyle="--")
    ax_uv.plot(t, uv_pred[:, 1], label="pred_v", color="tab:red", linestyle="--")
    ax_uv.set_ylabel("UV")
    ax_uv.set_title("UV Completion")
    ax_uv.legend(loc="upper right", ncols=4, fontsize=8)
    ax_uv.grid(alpha=0.3)

    ax_3d = axes[1]
    ax_3d.plot(t, pos3d[:, 0], label="x", color="tab:green")
    ax_3d.plot(t, pos3d[:, 1], label="y", color="tab:purple")
    ax_3d.plot(t, pos3d[:, 2], label="z", color="tab:brown")
    ax_3d.set_ylabel("3D")
    ax_3d.set_title("3D Trajectory")
    ax_3d.legend(loc="upper right", ncols=3, fontsize=8)
    ax_3d.grid(alpha=0.3)

    ax_evt = axes[2]
    for idx in range(probs.shape[1]):
        label = outputs.get("event_names", [f"event_{i}" for i in range(probs.shape[1])])[idx]
        ax_evt.plot(t, probs[:, idx], label=label)
    peaks = outputs.get("event_peaks", [])
    peak_scores = outputs.get("event_peak_scores", [])
    for event_idx, event_peaks in enumerate(peaks):
        if event_idx >= len(peak_scores):
            continue
        for peak_idx, peak_t in enumerate(event_peaks):
            score = peak_scores[event_idx][peak_idx]
            ax_evt.scatter([peak_t], [score], s=30, marker="x", color="black")
    ax_evt.set_ylabel("Probability")
    ax_evt.set_xlabel("Frame")
    ax_evt.set_ylim(0.0, 1.05)
    ax_evt.set_title("Event Timeline")
    ax_evt.legend(loc="upper right", ncols=max(1, probs.shape[1]), fontsize=8)
    ax_evt.grid(alpha=0.3)

    plt.tight_layout()
    return fig
