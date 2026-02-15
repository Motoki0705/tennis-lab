"""Timeline rendering helpers for event detection visualization."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _annotate_peaks(
    ax: Axes,
    *,
    peaks: list[int],
    scores: list[float],
    y: np.ndarray,
    color: str,
) -> None:
    for idx, score in zip(peaks, scores, strict=False):
        if 0 <= idx < y.shape[0]:
            ax.text(
                idx,
                float(y[idx]) + 0.03,
                f"{score:.2f}",
                fontsize=7,
                color=color,
                ha="center",
                va="bottom",
            )


def render_timeline_axes(
    axes: list[Axes],
    *,
    threshold: float,
    targets: np.ndarray,
    shot_indices: list[int],
    bounce_indices: list[int],
    probs: np.ndarray | None = None,
    pred_peaks: list[list[int]] | None = None,
    pred_scores: list[list[float]] | None = None,
    event_names: list[str] | None = None,
) -> list[str]:
    """Render per-event timeline plots onto pre-created axes."""
    _, num_events = targets.shape
    if len(axes) != num_events:
        raise ValueError(f"Expected {num_events} axes, got {len(axes)}")

    names = event_names or ["shot", "bounce"][:num_events]
    if len(names) < num_events:
        names = names + [f"event_{i}" for i in range(len(names), num_events)]

    x = np.arange(targets.shape[0])
    for event_idx in range(num_events):
        ax = axes[event_idx]
        ax.plot(
            x,
            targets[:, event_idx],
            color="white",
            alpha=0.6,
            linestyle="--",
            label="GT target",
        )

        if probs is not None:
            ax.plot(
                x,
                probs[:, event_idx],
                color="#00D1FF",
                alpha=0.9,
                label="Pred prob",
            )

        ax.axhline(
            threshold,
            color="yellow",
            alpha=0.5,
            linestyle="--",
            linewidth=1.2,
        )

        gt_idx = shot_indices if event_idx == 0 else bounce_indices if event_idx == 1 else []
        for frame in gt_idx:
            ax.axvline(frame, color="lime", alpha=0.25)

        if pred_peaks is not None and event_idx < len(pred_peaks):
            for frame in pred_peaks[event_idx]:
                ax.axvline(frame, color="magenta", alpha=0.25)

        if (
            probs is not None
            and pred_peaks is not None
            and pred_scores is not None
            and event_idx < len(pred_peaks)
        ):
            _annotate_peaks(
                ax,
                peaks=pred_peaks[event_idx],
                scores=pred_scores[event_idx] if event_idx < len(pred_scores) else [],
                y=probs[:, event_idx],
                color="magenta",
            )

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(names[event_idx])
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frame")
    return list(names)


def create_timeline_figure(
    *,
    threshold: float,
    targets: np.ndarray,
    shot_indices: list[int],
    bounce_indices: list[int],
    scene_id: str,
    title_suffix: str,
    probs: np.ndarray | None = None,
    pred_peaks: list[list[int]] | None = None,
    pred_scores: list[list[float]] | None = None,
    event_names: list[str] | None = None,
) -> Figure:
    """Create a per-event timeline plot."""
    _, num_events = targets.shape

    fig, axes_raw = plt.subplots(num_events, 1, figsize=(14, 2.6 * num_events), sharex=True)
    axes: list[Axes] = [cast(Axes, axes_raw)] if num_events == 1 else [cast(Axes, a) for a in axes_raw]

    render_timeline_axes(
        axes,
        threshold=threshold,
        targets=targets,
        shot_indices=shot_indices,
        bounce_indices=bounce_indices,
        probs=probs,
        pred_peaks=pred_peaks,
        pred_scores=pred_scores,
        event_names=event_names,
    )

    fig.suptitle(f"Event timeline | scene={scene_id}{title_suffix}")
    plt.tight_layout()
    return fig
