"""Reporting and persistence helpers for ball multitask visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.experiments.ball_multitask.visualization.types import SceneInputs


def save_outputs(output_path: Path, outputs: dict[str, Any]) -> None:
    """Persist inference outputs as NPZ + JSON event peaks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        uv_completed=outputs["uv_completed"],
        position_3d=outputs["position_3d"],
        event_logits=outputs["event_logits"],
        event_probs=outputs["event_probs"],
        in_frame_logits=outputs["in_frame_logits"],
        in_frame_probs=outputs["in_frame_probs"],
        in_frame_pred=outputs["in_frame_pred"],
    )
    peaks_path = output_path.with_suffix(".events.json")
    payload = {
        "event_names": outputs["event_names"],
        "event_peaks": outputs["event_peaks"],
        "event_peak_scores": outputs["event_peak_scores"],
    }
    peaks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_figure(path: Path, fig: plt.Figure) -> None:
    """Save figure to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")


def save_animation(path: Path, anim: FuncAnimation, *, fps: float) -> None:
    """Save one animation to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), fps=float(fps))


def print_info(scene: SceneInputs) -> None:
    """Print scene input information."""
    print(f"scene length: {scene.seq_len}")
    print(f"camera index: {scene.camera_idx}")
    print(f"ball_uv shape: {tuple(scene.ball_uv.shape)}")
    print(f"court_kp shape: {tuple(scene.court_kp.shape)}")


def print_predict_summary(outputs: dict[str, Any]) -> None:
    """Print concise prediction summary."""
    print(f"uv_completed: {tuple(outputs['uv_completed'].shape)}")
    print(f"position_3d: {tuple(outputs['position_3d'].shape)}")
    print(f"event_logits: {tuple(outputs['event_logits'].shape)}")
    print(f"in_frame_logits: {tuple(outputs['in_frame_logits'].shape)}")
    print(f"event peaks: {[len(v) for v in outputs['event_peaks']]}")
