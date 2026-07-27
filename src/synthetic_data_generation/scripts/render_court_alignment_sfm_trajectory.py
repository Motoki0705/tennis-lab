"""Render an animated GIF overlaying 3DGS court alignments with a dual-court 3D SfM navigation panel.

Usage:
    python -m src.synthetic_data_generation.scripts.render_court_alignment_sfm_trajectory

Notes:
    - Uses Hydra for configuration loading from ``src/synthetic_data_generation/configs/render_court_alignment_sfm_trajectory.yaml``.
    - Visualizes main court (Court-0) in Cyan and adjacent court (Court-1) in Gold alongside the SfM camera trajectory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from omegaconf import DictConfig
from PIL import Image

from src.synthetic_data_generation.alignment.court_template_fit import (
    court_line_segments,
)
from src.synthetic_data_generation.alignment.view_inputs import (
    load_provider_rgb_image,
)
from src.synthetic_data_generation.provider.bundle import (
    load_scene_provider_bundle,
)


def _render_single_frame(
    cam: Any,
    bundle: Any,
    c0_mat: np.ndarray,
    c1_mat: np.ndarray,
    cams: Sequence[Any],
    segs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    figsize: tuple[int, int],
    dpi: int,
) -> Image.Image:
    """Render a side-by-side 2D overlay and 3D navigation frame."""
    img_rgb = load_provider_rgb_image(bundle.image_path(cam.camera_id))

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#0B0F19")

    # Left Panel: RGB + Overlay
    ax1 = fig.add_axes((0.02, 0.05, 0.47, 0.88))
    ax1.imshow(img_rgb)
    ax1.axis("off")

    c2s = np.array(cam.camera_to_scene).reshape(4, 4)
    s2c = np.linalg.inv(c2s)
    K = np.array(cam.intrinsics).reshape(3, 3)

    def plot_court_overlay_clean(
        ax: Any, c_mat: np.ndarray, color: str, glow_color: str, label: str
    ) -> None:
        first_legend = True
        for p1, p2 in segs:
            pts = np.linspace(p1, p2, 80)
            hom = np.column_stack([pts, np.zeros(len(pts)), np.ones(len(pts))])
            scene_pts = (c_mat @ hom.T).T
            cam_pts = (s2c @ scene_pts.T).T
            valid = cam_pts[:, 2] > 0.05
            if not np.any(valid):
                continue
            uv = (K @ cam_pts[:, :3].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            in_b = (
                valid
                & (uv[:, 0] >= 0)
                & (uv[:, 0] <= cam.width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] <= cam.height)
            )

            for i in range(len(pts) - 1):
                if in_b[i] and in_b[i + 1]:
                    lbl = label if first_legend else None
                    if first_legend:
                        first_legend = False
                    ax.plot(
                        uv[i : i + 2, 0],
                        uv[i : i + 2, 1],
                        color=glow_color,
                        linewidth=5.0,
                        alpha=0.35,
                        solid_capstyle="round",
                    )
                    ax.plot(
                        uv[i : i + 2, 0],
                        uv[i : i + 2, 1],
                        color=color,
                        linewidth=2.0,
                        alpha=0.95,
                        label=lbl,
                    )

    plot_court_overlay_clean(ax1, c0_mat, "#00F2FE", "#00F2FE", "Court-0 (Main)")
    plot_court_overlay_clean(ax1, c1_mat, "#FFB300", "#FFB300", "Court-1 (Adjacent)")

    hud_text = (
        f"SCENE: b00-default-v1 (3DGS Synthetic)\n"
        f"CAMERA: {cam.camera_id} | GROUP: {cam.group_id}\n"
        f"ALIGNMENT: Court-0 (Cyan) / Court-1 (Gold)"
    )
    ax1.text(
        0.03,
        0.96,
        hud_text,
        transform=ax1.transAxes,
        color="white",
        fontsize=9,
        fontweight="bold",
        family="monospace",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#0B0F19",
            alpha=0.85,
            edgecolor="#00F2FE",
        ),
    )
    ax1.legend(
        loc="lower right",
        facecolor="#0B0F19",
        edgecolor="#475569",
        labelcolor="white",
        fontsize=8,
    )
    ax1.set_title(
        "3DGS Render & Dual-Court Alignment Overlay",
        color="white",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # Right Panel: 3D Scene / 2-Court Navigation
    ax2: Any = fig.add_axes((0.51, 0.05, 0.47, 0.88), projection="3d", facecolor="#0B0F19")
    ax2.set_title(
        "2-Court Space & SfM Camera Trajectory (3D)",
        color="white",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#1e293b")
    ax2.grid(True, color="#1e293b", linestyle="--", linewidth=0.5)
    ax2.tick_params(colors="#64748b", labelsize=8)

    def plot_court_3d(
        ax: Any, c_mat: np.ndarray, line_color: str, surf_color: str, name: str
    ) -> None:
        for p1, p2 in segs:
            pts = np.array([p1, p2])
            hom = np.column_stack([pts, np.zeros(len(pts)), np.ones(len(pts))])
            sc = (c_mat @ hom.T).T
            ax.plot(
                sc[:, 0], sc[:, 1], sc[:, 2], color=line_color, linewidth=1.5, alpha=0.95
            )
        corners = np.array(
            [
                [-5.485, -11.885, 0],
                [5.485, -11.885, 0],
                [5.485, 11.885, 0],
                [-5.485, 11.885, 0],
            ]
        )
        hom_c = np.column_stack([corners, np.ones(4)])
        sc_c = (c_mat @ hom_c.T).T[:, :3]
        poly = Poly3DCollection(
            [sc_c], facecolors=surf_color, edgecolors="none", alpha=0.35
        )
        ax.add_collection3d(poly)
        center = (c_mat @ [0, 0, 0, 1])[:3]
        ax.text(
            center[0],
            center[1],
            center[2] + 0.05,
            name,
            color=line_color,
            fontweight="bold",
            fontsize=9,
            ha="center",
        )

    plot_court_3d(ax2, c0_mat, "#00F2FE", "#064e3b", "Court-0")
    plot_court_3d(ax2, c1_mat, "#FFB300", "#78350f", "Court-1")

    all_pos = np.array(
        [np.array(c.camera_to_scene).reshape(4, 4)[:3, 3] for c in cams]
    )
    ax2.plot(
        all_pos[:, 0],
        all_pos[:, 1],
        all_pos[:, 2],
        color="#64748b",
        linewidth=1.2,
        alpha=0.5,
        label="SfM Trajectory",
    )

    curr_pos = c2s[:3, 3]
    ax2.scatter(
        xs=curr_pos[0],
        ys=curr_pos[1],
        zs=curr_pos[2],
        color="#FF007F",
        s=90,
        marker="o",
        label="Current Camera",
        zorder=10,
    )
    fwd = c2s[:3, 2] * 0.35
    ax2.quiver(
        curr_pos[0],
        curr_pos[1],
        curr_pos[2],
        fwd[0],
        fwd[1],
        fwd[2],
        color="#FF007F",
        linewidth=2.0,
        arrow_length_ratio=0.3,
    )

    ax2.set_xlabel("X (m)", color="#64748b")
    ax2.set_ylabel("Y (m)", color="#64748b")
    ax2.set_zlabel("Z (m)", color="#64748b")
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_zlim(-0.3, 0.15)
    ax2.legend(
        loc="upper right",
        facecolor="#0B0F19",
        edgecolor="#475569",
        labelcolor="white",
        fontsize=8,
    )
    ax2.view_init(elev=35, azim=-55)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img_array = rgba[:, :, :3]
    plt.close(fig)

    return Image.fromarray(img_array)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="render_court_alignment_sfm_trajectory",
)
def main(config: DictConfig) -> None:
    """Render animated GIF from SfM camera trajectory and dual-court alignment."""
    scene_path = Path(config.scene_dir)
    geom_path = Path(config.court_geometry_path)
    output_path = Path(config.output_gif_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_scene_provider_bundle(scene_path, verify_files=False)
    with open(geom_path, encoding="utf-8") as f:
        geom = json.load(f)

    c0_mat = np.array(geom["candidates"][0]["scene_from_court"]).reshape(4, 4)
    c1_mat = np.array(geom["candidates"][1]["scene_from_court"]).reshape(4, 4)
    cams = bundle.manifest.cameras
    segs = court_line_segments()

    step = int(config.step)
    selected_cams = cams[::step]
    print(f"Rendering {len(selected_cams)} frames for {output_path} ...")

    frames: list[Image.Image] = []
    figsize = tuple(config.figsize)
    dpi = int(config.dpi)

    for idx, cam in enumerate(selected_cams):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Frame {idx + 1}/{len(selected_cams)} (camera {cam.camera_id})")
        frame = _render_single_frame(
            cam=cam,
            bundle=bundle,
            c0_mat=c0_mat,
            c1_mat=c1_mat,
            cams=cams,
            segs=segs,
            figsize=figsize,
            dpi=dpi,
        )
        frames.append(frame)

    if frames:
        duration_ms = int(1000 / int(config.fps))
        print(f"Saving GIF to {output_path} (duration: {duration_ms}ms/frame) ...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0,
        )
        print("GIF rendering complete!")


if __name__ == "__main__":
    main()
