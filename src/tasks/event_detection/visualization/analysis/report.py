"""Reporting and output helpers for event detection visualization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from src.event_detection.visualization.types import Traj3DEventInputs, UVEventInputs


def print_uv_info(scene_path: Path, inputs: UVEventInputs) -> None:
    """Print compact UV scene info."""
    scene_id = inputs.meta.get("scene_id", "Unknown")
    print("=" * 60)
    print("EVENT_DETECTION UV VISUALIZATION")
    print("=" * 60)
    print(f"Scene:   {scene_id}")
    print(f"Path:    {scene_path}")
    print(f"Camera:  {inputs.camera_idx}")
    print(f"Frames:  {inputs.ball_uv.shape[0]}")
    print(f"GT shot indices:   {inputs.shot_indices}")
    print(f"GT bounce indices: {inputs.bounce_indices}")


def print_traj3d_info(scene_path: Path, inputs: Traj3DEventInputs) -> None:
    """Print compact 3D scene info."""
    scene_id = inputs.meta.get("scene_id", "Unknown")
    print("=" * 60)
    print("EVENT_DETECTION 3D VISUALIZATION")
    print("=" * 60)
    print(f"Scene:   {scene_id}")
    print(f"Path:    {scene_path}")
    print(f"Frames:  {inputs.ball_pos_world.shape[0]}")
    print(f"GT shot indices:   {inputs.shot_indices}")
    print(f"GT bounce indices: {inputs.bounce_indices}")


def save_figure(fig: Figure, path: Path) -> None:
    """Save matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")


def save_animation(anim: FuncAnimation, path: Path, fps: float) -> None:
    """Save matplotlib animation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), fps=float(fps))


def save_outputs(outputs: dict[str, object], output_path: Path) -> None:
    """Save prediction outputs as ``.pt`` or ``.json``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".pt":
        torch.save(outputs, output_path)
        return

    if output_path.suffix == ".json":
        json_data: dict[str, object] = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                json_data[key] = value.squeeze(0).cpu().tolist()
            else:
                json_data[key] = value
        output_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    raise ValueError(
        f"Unsupported output format: {output_path.suffix} (expected .pt or .json)"
    )
