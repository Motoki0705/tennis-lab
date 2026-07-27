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
from src.utils.rendering.camera_view import CameraView3D, apply_scene_camera


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
    ax1.set_xlim(0, cam.width)
    ax1.set_ylim(cam.height, 0)
    ax1.autoscale(False)
    ax1.axis("off")

    c2s = np.array(cam.camera_to_scene).reshape(4, 4)
    s2c = np.linalg.inv(c2s)
    K = np.array(cam.intrinsics).reshape(3, 3)

    def plot_court_overlay_clean(
        ax: Any, c_mat: np.ndarray, color: str, glow_color: str
    ) -> None:
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
                    )

    plot_court_overlay_clean(ax1, c0_mat, "#00F2FE", "#00F2FE")
    plot_court_overlay_clean(ax1, c1_mat, "#FFB300", "#FFB300")

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
    ax1.set_xlim(0, cam.width)
    ax1.set_ylim(cam.height, 0)
    ax1.set_title(
        "3DGS Render & Dual-Court Alignment Overlay",
        color="white",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # Right Panel: Rich 3D Dual-Court Space (PLCS/BLCS Base Style & Broadcast Look)
    ax2: Any = fig.add_axes((0.51, 0.05, 0.47, 0.88), projection="3d", facecolor="#101418")
    ax2.set_title(
        "2-Court 3D Alignment & SfM Trajectory (BLCS/PLCS Style)",
        color="#E8E8E8",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # Completely remove matplotlib axes chrome for the broadcast look
    ax2.set_axis_off()

    def plot_rich_court_3d(
        ax: Any,
        c_mat: np.ndarray,
        court_color: str,
        apron_color: str,
        line_color: str,
        net_color: str,
        name: str,
    ) -> None:
        # 1. Outer apron surface (run-off area)
        apron = np.array(
            [
                [-7.315, -15.0, -0.005],
                [7.315, -15.0, -0.005],
                [7.315, 15.0, -0.005],
                [-7.315, 15.0, -0.005],
            ]
        )
        hom_a = np.column_stack([apron, np.ones(4)])
        sc_a = (c_mat @ hom_a.T).T[:, :3]
        poly_a = Poly3DCollection(
            [sc_a], facecolors=apron_color, edgecolors="none", alpha=0.9
        )
        ax.add_collection3d(poly_a)

        # 2. Main playing court surface
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
        poly_c = Poly3DCollection(
            [sc_c], facecolors=court_color, edgecolors="none", alpha=0.95
        )
        ax.add_collection3d(poly_c)

        # 3. Court lines (clean white, standard width)
        for p1, p2 in segs:
            pts = np.array([p1, p2])
            hom = np.column_stack([pts, np.zeros(len(pts)), np.ones(len(pts))])
            sc = (c_mat @ hom.T).T
            ax.plot(
                sc[:, 0],
                sc[:, 1],
                sc[:, 2],
                color=line_color,
                linewidth=2.0,
                alpha=0.95,
            )

        # 4. Realistic 3D Net, Posts, and Mesh
        x_net = np.linspace(-5.485, 5.485, 25)
        z_net = 0.914 + (1.07 - 0.914) * (x_net / 5.485) ** 2
        net_top = np.column_stack([x_net, np.zeros_like(x_net), z_net, np.ones_like(x_net)])
        sc_net_top = (c_mat @ net_top.T).T
        ax.plot(
            sc_net_top[:, 0],
            sc_net_top[:, 1],
            sc_net_top[:, 2],
            color="#FFFFFF",
            linewidth=2.5,
            alpha=0.95,
        )

        for sx in [-5.485, 5.485]:
            post = np.array([[sx, 0, 0, 1], [sx, 0, 1.07, 1]])
            sc_post = (c_mat @ post.T).T
            ax.plot(
                sc_post[:, 0],
                sc_post[:, 1],
                sc_post[:, 2],
                color="#cbd5e1",
                linewidth=2.8,
                alpha=0.9,
            )

        for hz in [0.3, 0.6]:
            net_mid = np.column_stack([x_net, np.zeros_like(x_net), np.full_like(x_net, hz), np.ones_like(x_net)])
            sc_net_mid = (c_mat @ net_mid.T).T
            ax.plot(
                sc_net_mid[:, 0],
                sc_net_mid[:, 1],
                sc_net_mid[:, 2],
                color=net_color,
                linewidth=0.6,
                alpha=0.45,
            )

        center_pt = (c_mat @ [0, 0, 0, 1])[:3]
        ax.text(
            center_pt[0],
            center_pt[1],
            center_pt[2] + 0.08,
            name,
            color="#FFFFFF",
            fontweight="bold",
            fontsize=9,
            ha="center",
        )

    # Transform into Court-0 centric standard coordinate system with 180-degree XY rotation
    # (prevents 180-degree front/back inversion relative to TV broadcast view)
    rot180 = np.diag([-1.0, -1.0, 1.0, 1.0])
    inv_c0 = rot180 @ np.linalg.inv(c0_mat)
    i_mat = np.eye(4)
    c1_rel = inv_c0 @ c1_mat


    # Court-0: Official DARK_THEME court colors (#4C9B57 court, #33763D apron)
    plot_rich_court_3d(
        ax2, i_mat, court_color="#4C9B57", apron_color="#33763D", line_color="#FFFFFF", net_color="#B9C0C7", name="Court-0 (Main)"
    )
    # Court-1: Slightly deeper tennis green tone (#3A8A45 court, #2A6632 apron)
    plot_rich_court_3d(
        ax2, c1_rel, court_color="#3A8A45", apron_color="#2A6632", line_color="#E8E8E8", net_color="#B9C0C7", name="Court-1 (Adjacent)"
    )

    all_pos = np.array(
        [(inv_c0 @ np.array(c.camera_to_scene).reshape(4, 4))[:3, 3] for c in cams]
    )
    ax2.plot(
        all_pos[:, 0],
        all_pos[:, 1],
        all_pos[:, 2],
        color="#94a3b8",
        linewidth=1.3,
        alpha=0.6,
        label="SfM Trajectory",
    )
    ax2.scatter(
        xs=all_pos[::4, 0],
        ys=all_pos[::4, 1],
        zs=all_pos[::4, 2],
        color="#38bdf8",
        s=8,
        alpha=0.4,
    )

    curr_pos = (inv_c0 @ c2s)[:3, 3]
    ax2.scatter(
        xs=curr_pos[0],
        ys=curr_pos[1],
        zs=curr_pos[2],
        color="#FF007F",
        s=250,
        marker="o",
        alpha=0.25,
    )
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

    # FOV Frustum pyramid in Court-0 centric coordinates
    w, h = cam.width, cam.height
    scale = 0.6
    inv_K = np.linalg.inv(K)
    corners_uv = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    rays = (inv_K @ corners_uv.T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    pts_cam = rays * scale
    pts_hom = np.column_stack([pts_cam, np.ones(4)])
    pts_world = (inv_c0 @ c2s @ pts_hom.T).T[:, :3]

    for pt in pts_world:
        ax2.plot(
            [curr_pos[0], pt[0]],
            [curr_pos[1], pt[1]],
            [curr_pos[2], pt[2]],
            color="#FF007F",
            linewidth=1.5,
            alpha=0.85,
        )
    w_loop = np.vstack([pts_world, pts_world[0]])
    ax2.plot(
        w_loop[:, 0],
        w_loop[:, 1],
        w_loop[:, 2],
        color="#FFB300",
        linewidth=1.5,
        alpha=0.9,
    )

    view = CameraView3D(elev=26.0, azim=-105.0, zoom=0.95)
    apply_scene_camera(ax2, view, margin=11.0, z_limit=4.0)
    ax2.legend(
        loc="upper right",
        facecolor="#101418",
        edgecolor="#334155",
        labelcolor="#E8E8E8",
        fontsize=8,
    )


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
