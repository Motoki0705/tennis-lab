"""Visualize measured SfM tracks, shared ground cells and temporal height bias."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, sha256, write_json
from drift_evidence import BUNDLE, collect, geometry, load_bundle, measure, validate
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from PIL import Image


def drift_figure() -> plt.Figure:
    manifest, tracks, points, method, alignment = load_bundle()
    measurement = measure(manifest, tracks, points, method, alignment)["primary"]
    geom = geometry(points, method, alignment)
    edges = manifest["period_edges"]
    delta = 100 * np.array(measurement["cell_delta_from_first_m"])
    medians = np.median(delta, axis=1)
    quartiles = np.quantile(delta, [0.25, 0.75], axis=1)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig = plt.figure(figsize=(10.5, 5.3), layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 2.65], hspace=0.12)
        for i, name in enumerate(manifest["photos"]):
            ax = fig.add_subplot(grid[0, i])
            ax.imshow(Image.open(BUNDLE / name))
            ax.set_title(f"Q{i + 1}: frames {edges[i]}–{edges[i + 1] - 1}", pad=4)
            ax.text(
                0.02,
                0.03,
                f"frame {edges[i]:06d}",
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.6,
                    "pad": 2,
                    "edgecolor": "none",
                },
            )
            ax.axis("off")
        ax = fig.add_subplot(grid[1, :2])
        # This layer is the actual ground ROI, not an idealized court drawing.
        visible = geom["inside"] & (np.abs(geom["height"]) < 0.3)
        shown = np.flatnonzero(visible)[::3]
        ax.scatter(
            *geom["uv"][shown].T,
            c=points[shown, 3:],
            s=0.5,
            alpha=0.28,
            rasterized=True,
            linewidths=0,
        )
        for outline in geom["outlines"]:
            ax.plot(*outline.T, color="#657882", lw=0.7)
        size = measurement["cell_size_m"]
        patches = [
            Rectangle(cell * size, size, size)
            for cell in np.array(measurement["shared_cell_indices"])
        ]
        collection = PatchCollection(
            patches, cmap="RdBu_r", edgecolors="#263844", linewidths=0.4
        )
        collection.set_array(delta[2])
        collection.set_clim(-12, 12)
        ax.add_collection(collection)
        fig.colorbar(
            collection, ax=ax, label="Q3 − Q1 height [cm]", shrink=0.83, pad=0.025
        )
        ax.set(
            title="(a) All 33 shared 0.5 m ground cells",
            xlabel="Ground U [m]",
            ylabel="Ground V [m]",
            aspect="equal",
            xticks=[-10, 0, 10],
            yticks=[-15, -10, -5, 0, 5, 10],
        )
        ax.margins(0.07)

        ax = fig.add_subplot(grid[1, 2:])
        periods = np.arange(1, 5)
        # Fixed offsets only separate coincident marks; each dot is a measured cell.
        offsets = np.linspace(-0.15, 0.15, delta.shape[1])
        for i in range(4):
            ax.scatter(
                i + 1 + offsets,
                delta[i],
                s=12,
                c="#8095a2",
                alpha=0.65,
                edgecolors="none",
                label="Individual shared cells" if i == 1 else None,
            )
        ax.axhline(0, color="#6d7d85", lw=0.8, ls="--")
        ax.errorbar(
            periods,
            medians,
            yerr=np.vstack([medians - quartiles[0], quartiles[1] - medians]),
            fmt="o",
            ms=6,
            color="#006b70",
            capsize=6,
            elinewidth=3,
            label="Median and interquartile range",
        )
        for x, median, q3 in zip(periods, medians, quartiles[1], strict=True):
            ax.text(
                x + 0.21,
                q3 + 0.75,
                f"{median:+.1f}",
                color="#006b70",
                fontsize=10,
                ha="center",
                weight="bold",
            )
        ax.set(
            title="(b) Same cells, four observation periods",
            ylabel="Cell height change from Q1 [cm]",
            xlabel="Observation frames (equal chronological quarters)",
            xticks=periods,
            xticklabels=[f"Q{i + 1}\n{edges[i]}–{edges[i + 1] - 1}" for i in range(4)],
            xlim=(0.6, 4.65),
            ylim=(-13, 13),
            yticks=[-10, -5, 0, 5, 10],
        )
        ax.grid(axis="y", alpha=0.15)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        # Tick creation may be deferred until savefig, outside rc_context.
        for axis in fig.axes:
            axis.tick_params(labelsize=9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    checks = validate(check_local_sources=args.collect)
    figure = drift_figure()
    name = "sfm_temporal_drift.png"
    figure.savefig(ROOT / "figures" / name, dpi=210, facecolor="white")
    plt.close(figure)
    receipt = {
        "bundle_manifest_sha256": sha256(BUNDLE / "manifest.json"),
        "measurements_sha256": sha256(BUNDLE / "measurements.json"),
        "figures": {name: sha256(ROOT / "figures" / name)},
        "verification": checks,
        "display": "Full captured frames at quarter starts; actual ground points, every third point for display only; all shared cells colored by Q3 minus Q1, fixed ±12 cm. All individual cell differences, median and IQR, no smoothing. No independent survey ground truth.",
    }
    write_json(ROOT / "evidence/drift_figures.json", receipt)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
