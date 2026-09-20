"""Plot measured RANSAC support and saved view-to-ground LINE inference."""

from __future__ import annotations

import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from alignment_evidence import (
    BUNDLE,
    SELECTED,
    aggregate,
    collect,
    ground_fit,
    raster_view,
    read_bundle,
    validate_geometry,
)
from common import ROOT, sha256, write_json
from matplotlib.colors import PowerNorm
from PIL import Image

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def ransac_figure(manifest: dict, arrays: dict) -> tuple[plt.Figure, dict]:
    plane, candidates, support = ground_fit(manifest, arrays)
    scale = manifest["nht_scene_units_per_metre"]
    xyz = arrays["points_xyzrgb"][:, :3].astype(np.float64) / scale
    colors = arrays["points_xyzrgb"][:, 3:]
    u0, u1, v0, v1 = plane.support_uv_bounds
    corners = plane.from_uv(np.array([[u0, v0], [u0, v1], [u1, v0], [u1, v1]])) / scale
    bounds = np.stack([corners.min(axis=0), corners.max(axis=0)])
    bounds[0] -= [2, 2, 1]
    bounds[1] += [2, 2, 6]
    visible = np.all((xyz >= bounds[0]) & (xyz <= bounds[1]), axis=1)
    indices = np.flatnonzero(visible)[::3]
    fig = plt.figure(figsize=(10.5, 3.0), layout="constrained")
    titles = [
        "(a) SfM points: measured RGB",
        "(b) Ground-height candidates",
        "(c) RANSAC + SVD ground plane",
    ]
    for i, title in enumerate(titles):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d", computed_zorder=False)
        ax.view_init(elev=36, azim=-63)
        ax.set_proj_type("ortho")
        if i == 0:
            ax.scatter(*xyz[indices].T, c=colors[indices], s=0.35, depthshade=False)
        else:
            ax.scatter(
                *xyz[indices].T,
                c="#aebdcc",
                s=0.25,
                alpha=0.08,
                depthshade=False,
                zorder=1,
            )
            mask = candidates if i == 1 else support
            chosen = indices[mask[indices]]
            ax.scatter(
                *xyz[chosen].T,
                c="#dc8229" if i == 1 else "#007f83",
                s=1,
                depthshade=False,
                zorder=5,
            )
            if i == 2:
                u0, u1, v0, v1 = plane.support_uv_bounds
                u, v = np.meshgrid(np.linspace(u0, u1, 6), np.linspace(v0, v1, 6))
                plane_xyz = plane.from_uv(np.stack([u, v], axis=-1)) / scale
                ax.plot_wireframe(
                    *plane_xyz.transpose(2, 0, 1),
                    color="#a42a51",
                    linewidth=0.85,
                    alpha=1,
                    zorder=10,
                )
        ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2], title=title)
        ax.set_box_aspect(np.maximum(bounds[1] - bounds[0], 0.1))
        ax.set_xlabel("X [m]", labelpad=-1)
        ax.set_ylabel("Y [m]", labelpad=-1)
        ax.set_zlabel("Z [m]", labelpad=-2)
        ax.tick_params(pad=0)
        ax.locator_params(nbins=3)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["color"] = (0.8, 0.85, 0.88, 0.3)
    return fig, {
        "display_bounds_metres": bounds.tolist(),
        "displayed_points": len(indices),
        "display_policy": "Common ground-support ROI expanded by 2 m horizontally, 1 m below and 6 m above; every third visible point. Candidate/support/plane layers are drawn in the foreground. Fitting uses all original points. Metric scale is applied only after fitting.",
    }


def probability_overlay(rgb: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = cv2.resize(
        probability, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    color = plt.get_cmap("inferno")(p)[..., :3]
    result = rgb.astype(np.float64) / 255 * 0.7
    active = p >= 0.5
    result[active] = result[active] * 0.15 + color[active] * 0.85
    return result


def display_grid(value: np.ndarray) -> np.ma.MaskedArray:
    """Preserve thin evidence at print size with disclosed 3x3 display bins."""
    h, w = value.shape
    padded = np.pad(value, ((0, -h % 3), (0, -w % 3)))
    reduced = padded.reshape(padded.shape[0] // 3, 3, padded.shape[1] // 3, 3).max(
        axis=(1, 3)
    )
    return np.ma.masked_equal(reduced, 0)


def projection_figure(manifest: dict, arrays: dict) -> tuple[plt.Figure, dict]:
    ids = arrays["camera_ids"].tolist()
    indices = [ids.index(i) for i in SELECTED]
    selected = aggregate(manifest, arrays, indices)
    fit = arrays["evidence_sum"]
    # One fixed range across views/aggregates; pooling affects display only.
    maximum = float(fit.max())
    norm = PowerNorm(gamma=1 / 3, vmin=0, vmax=maximum)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    u0, _, v0, _ = manifest["bounds_uv"]
    h, w = fit.shape
    h, w = 3 * ((h + 2) // 3), 3 * ((w + 2) // 3)
    spacing = manifest["grid_spacing"]
    extent = [
        u0 - spacing / 2,
        u0 + (w - 0.5) * spacing,
        v0 - spacing / 2,
        v0 + (h - 0.5) * spacing,
    ]
    fig = plt.figure(figsize=(10.5, 7.6), layout="constrained")
    grid = fig.add_gridspec(
        3, 4, width_ratios=[1, 1, 1, 0.035], height_ratios=[0.62, 1, 1]
    )
    for col, (camera_id, index) in enumerate(zip(SELECTED, indices, strict=True)):
        ax = fig.add_subplot(grid[0, col])
        rgb = np.asarray(Image.open(BUNDLE / f"{camera_id}.png").convert("RGB"))
        ax.imshow(probability_overlay(rgb, arrays[f"probability_{camera_id}"]))
        ax.set_title(f"{camera_id}: RGB + LINE p >= 0.5")
        ax.axis("off")
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(
            display_grid(raster_view(manifest, arrays, index)),
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(f"Ground projection: view {col + 1}")
        ax.set(xlabel="u [m]", ylabel="v [m]")
    scalar = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0, 1), cmap="inferno"
    )
    fig.colorbar(scalar, cax=fig.add_subplot(grid[0, 3]), label="LINE probability p")
    final_bundle_path = ROOT / "evidence/scene_sources/B00.json"
    final_bundle = json.loads(final_bundle_path.read_text())
    for col, (value, title) in enumerate(
        [
            (selected, "Sum of these 3 fit views"),
            (fit, "Sum of all 32 fit views"),
            (fit, "32-view sum + saved court alignment"),
        ]
    ):
        ax = fig.add_subplot(grid[2, col])
        artist = ax.imshow(
            display_grid(value),
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="u [m]", ylabel="v [m]")
        if col == 2:
            frame = manifest["ground_plane_frame"]
            origin = np.asarray(frame["origin_metric_scene"])
            basis = np.stack(
                [frame["basis_u_metric_scene"], frame["basis_v_metric_scene"]], axis=1
            )
            local = np.asarray(final_bundle["local_keypoints_3d"])
            for court in final_bundle["alignment"]["layout"]["courts"]:
                transform = np.asarray(court["scene_from_court"]).reshape(4, 4)
                uv = (local @ transform[:3, :3].T + transform[:3, 3] - origin) @ basis
                for a, b in final_bundle["segments"]:
                    ax.plot(
                        uv[[a, b], 0], uv[[a, b], 1], color="#cb3d75", linewidth=0.85
                    )
    fig.colorbar(
        artist,
        cax=fig.add_subplot(grid[1:, 3]),
        label="Weighted evidence E (display gamma = 1/3)",
    )
    return fig, {
        "ground_range": [0, maximum],
        "ground_display_gamma": 1 / 3,
        "ground_interpolation": "3x3 maximum display bins, nearest interpolation; original numeric arrays unchanged",
        "ground_zero": "white; no projected support",
        "raw_probability_range": [0, 1],
        "raw_overlay_threshold": 0.5,
        "raw_overlay": "0.7*RGB background; inferno color at LINE probability >=0.5",
        "alignment_bundle_sha256": sha256(final_bundle_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    manifest, arrays = read_bundle()
    verified = validate_geometry(manifest, arrays)
    figures = {}
    policies = {}
    for name, builder in [
        ("method_ransac", ransac_figure),
        ("method_projection", projection_figure),
    ]:
        fig, policy = builder(manifest, arrays)
        path = ROOT / "figures" / f"{name}.png"
        fig.savefig(
            path,
            dpi=210,
            facecolor="white",
            metadata={"Software": "court_robustness/make_alignment_figures.py"},
        )
        plt.close(fig)
        figures[path.name] = sha256(path)
        policies[name] = policy
    write_json(
        ROOT / "evidence/alignment_figures.json",
        {
            "bundle_manifest_sha256": sha256(BUNDLE / "manifest.json"),
            "figures": figures,
            "display": policies,
            "verification": verified,
        },
    )
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
