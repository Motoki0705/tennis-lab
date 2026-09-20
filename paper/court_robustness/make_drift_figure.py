"""Visualize measured SfM tracks, shared ground cells and temporal height bias."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT, sha256, write_json
from drift_evidence import (
    BUNDLE,
    SCENE_IDS,
    collect,
    geometry,
    load_bundle,
    measure,
    validate,
)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


def drift_figure() -> plt.Figure:
    """Every shared cell, with identical metric color limits for all scenes."""
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.7))
        fig.subplots_adjust(
            left=0.06, right=0.88, bottom=0.08, top=0.9, hspace=0.5, wspace=0.2
        )
        for sid, ax in zip(SCENE_IDS, axes.flat, strict=True):
            manifest, tracks, points, reference, alignment = load_bundle(sid)
            measurement = measure(manifest, tracks, points, reference, alignment)[
                "primary"
            ]
            geom = geometry(points, reference, alignment)
            visible = geom["inside"] & (
                np.abs(geom["height"])
                < manifest["selection"]["maximum_absolute_height_m"]
            )
            shown = np.flatnonzero(visible)[::3]
            ax.scatter(
                *geom["uv"][shown].T,
                c=points[shown, 3:],
                s=0.5,
                alpha=0.25,
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
                patches, cmap="RdBu_r", edgecolors="#263844", linewidths=0.25
            )
            difference = 100 * np.array(measurement["cell_delta_from_first_m"])[1]
            if np.abs(difference).max() > 20:
                raise ValueError("Height difference exceeds the shared color scale")
            collection.set_array(difference)
            collection.set_clim(-20, 20)
            ax.add_collection(collection)
            median = 100 * measurement["median_delta_m"][1]
            edges = manifest["period_edges"]
            ax.set(
                title=f"{sid} | n={measurement['shared_cell_count']} | median {median:+.2f} cm\nframes {edges[0]}–{edges[1] - 1} / {edges[1]}–{edges[2] - 1}",
                xlabel="Ground U [m]",
                ylabel="Ground V [m]",
                aspect="equal",
            )
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.margins(0.05)
        fig.colorbar(
            collection,
            cax=fig.add_axes([0.91, 0.16, 0.017, 0.67]),
            label="Later − earlier cell median height [cm]",
            ticks=[-20, -10, 0, 10, 20],
        )
        for axis in fig.axes:
            axis.tick_params(labelsize=8)
    return fig


def drift_table() -> str:
    """Typeset the measured numbers directly; do not manually transcribe values."""
    rows = [
        r"\begin{tabular}{@{}lrrrrll@{}}\toprule",
        r" & SfM画像 & SfM点 & 共通セル & 中央値 [cm] & 四分位範囲 [cm] & 1 m格子 [cm]（セル）\\\midrule",
    ]
    for sid in SCENE_IDS:
        bundle = load_bundle(sid)
        manifest = bundle[0]
        result = measure(*bundle)
        primary, sensitivity = result["primary"], result["sensitivity"]
        low, high = 100 * np.array(primary["iqr_delta_m"])[1]
        rows.append(
            f"{sid} & {manifest['frame_count']} & {manifest['point_count']:,} & "
            f"{primary['shared_cell_count']} & {100 * primary['median_delta_m'][1]:+.2f} & "
            f"[{low:+.2f}, {high:+.2f}] & {100 * sensitivity['median_delta_m'][1]:+.2f} ({sensitivity['shared_cell_count']})"
            + r"\\"
        )
    rows.extend([r"\bottomrule\end{tabular}", ""])
    return "\n".join(rows)


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
    table = ROOT / "tables/sfm_drift.tex"
    table.parent.mkdir(exist_ok=True)
    table.write_text(drift_table())
    receipt = {
        "bundle_manifest_sha256": sha256(BUNDLE / "manifest.json"),
        "measurements_sha256": sha256(BUNDLE / "measurements.json"),
        "figures": {name: sha256(ROOT / "figures" / name)},
        "tables": {str(table.relative_to(ROOT)): sha256(table)},
        "verification": checks,
        "display": "All four scenes, actual RGB ground points (every third for display only), saved court outlines, and every supported 0.5 m cell colored by later minus earlier median height. Shared color limits ±20 cm encompass all measured cell differences; no smoothing or data clipping. No survey ground truth.",
    }
    write_json(ROOT / "evidence/drift_figures.json", receipt)
    print(drift_table())


if __name__ == "__main__":
    main()
