"""Standalone diagnostic plots and a synchronized court skeleton comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from numpy.typing import NDArray

from src.utils.schema.player import COCO17_SKELETON


def plot_diagnostics(output_dir: Path, arrays: dict[str, Any], fps: float) -> None:
    time = np.arange(arrays["target_position"].shape[1]) / fps
    for player in range(len(arrays["target_position"])):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for key, label, color in [
            ("target", "PLCS", "black"),
            ("fixed", "GVHMR: scale=1", "#2878b5"),
            ("free", "GVHMR: fitted scale", "#d35d21"),
        ]:
            position = arrays[f"{key}_position"][player]
            axes[0, 0].plot(position[:, 0], position[:, 1], label=label, color=color)
            for axis in range(3):
                axes[1, axis].plot(time, position[:, axis], label=label, color=color)
            yaw = arrays[f"{key}_yaw"][player]
            axes[0, 1].plot(time, np.rad2deg(np.unwrap(yaw)), label=label, color=color)
            if key != "target":
                error = np.linalg.norm(
                    position - arrays["target_position"][player], axis=-1
                )
                axes[0, 2].plot(time, error, label=label, color=color)
        for ax, title in zip(
            axes.flat,
            [
                "Court XY (m)",
                "Unwrapped heading (deg)",
                "Distance to PLCS (m)",
                "X (m)",
                "Y (m)",
                "Z (m)",
            ],
            strict=True,
        ):
            ax.set_title(title)
            ax.grid(alpha=0.2)
        axes[0, 0].set_aspect("equal", adjustable="datalim")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Player {player}: one constant transform over the full track; PLCS is not ground truth"
        )
        fig.savefig(output_dir / f"player_{player}_diagnostics.png", dpi=150)
        plt.close(fig)


def render_comparison(
    output_dir: Path, arrays: dict[str, Any], fps: float, stride: int = 4
) -> None:
    """Render same source articulation in the existing and fitted placements."""
    joints: list[NDArray[np.float64]] = [
        arrays[f"{key}_joints"] for key in ("direct", "fixed", "free")
    ]
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    titles = [
        "Per-frame PLCS placement",
        "GVHMR world / scale = 1",
        "GVHMR world / fitted scale",
    ]
    colors = ["#e26925", "#206db3"]
    lines: list[Any] = []
    edges = np.asarray(COCO17_SKELETON)
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title)
        ax.set(
            xlim=(-6, 6), ylim=(-14, 14), zlim=(-1, 3), xlabel="X (m)", ylabel="Y (m)"
        )
        ax.set_box_aspect((12, 28, 8))
        ax.view_init(elev=32, azim=-55)
        for x in (-5.485, 5.485, -4.115, 4.115):
            ax.plot([x, x], [-11.885, 11.885], [0, 0], color="#777777", lw=0.8)
        for y in (-11.885, -6.4, 0, 6.4, 11.885):
            ax.plot([-5.485, 5.485], [y, y], [0, 0], color="#777777", lw=0.8)
        ax.plot([-5.485, 5.485], [0, 0], [0.914, 0.914], color="gray", lw=1)
        panel = []
        for player in range(joints[0].shape[0]):
            panel.append(
                [
                    ax.plot([], [], [], color=colors[player % len(colors)], lw=2)[0]
                    for _ in edges
                ]
            )
        lines.append(panel)
    writer = FFMpegWriter(
        fps=cast(Any, fps / stride),
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "23"],
    )
    frame_count = joints[0].shape[1]
    with writer.saving(fig, str(output_dir / "comparison.mp4"), dpi=90):
        for frame in range(0, frame_count, stride):
            for method, panel in enumerate(lines):
                for player, body_lines in enumerate(panel):
                    for line, edge in zip(body_lines, edges, strict=True):
                        xyz = joints[method][player, frame, edge]
                        line.set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            fig.suptitle(
                f"{frame / fps:.2f}s | shared input articulation; no temporal smoothing | feet below Z=0 remain visible"
            )
            if frame == 0:
                fig.savefig(output_dir / "comparison.png", dpi=130)
            writer.grab_frame()
    plt.close(fig)
