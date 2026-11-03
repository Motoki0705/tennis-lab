"""Small diagnostic plots that supplement the foundation comparison report."""

from __future__ import annotations

import base64
import html
import json

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def diagnostic_plots(directory, inputs, decode, candidates, smoothing):
    output = []
    figure = Figure(figsize=(14, 9), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3)
    width, height = inputs.source_size_wh
    for row, camera in enumerate(inputs.camera_ids):
        for column, segment in enumerate(("tuning", "evaluation", "all")):
            axis = axes[row, column]
            grid = decode[segment][camera]["all"]["position_grid_6x4"]
            cells = [c for c in grid if c["count"]]
            x = [(c["cell_xy"][0] + 0.5) * width / 6 for c in cells]
            y = [(c["cell_xy"][1] + 0.5) * height / 4 for c in cells]
            dx = [c["mean_signed_xy_px"][0] for c in cells]
            dy = [c["mean_signed_xy_px"][1] for c in cells]
            axis.quiver(
                x,
                y,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="#bb4f32",
                width=0.006,
            )
            for px, py, cell in zip(x, y, cells, strict=True):
                axis.text(px, py - 35, f"n={cell['count']}", fontsize=6, ha="center")
            axis.set(
                xlim=(0, width),
                ylim=(height, 0),
                title=f"{camera} / {segment}",
                xlabel="Reference u (native px)",
                ylabel="Reference v (native px)",
            )
            axis.grid(alpha=0.2)
    figure.suptitle(
        "Pre-gate log-parabola error: prediction - reference; 6x4 cell means (1:1 px vectors)"
    )
    figure.tight_layout()
    path = directory / "position-bias.png"
    figure.savefig(path)
    figure.clear()
    output.append(path)

    figure = Figure(figsize=(13, 4), dpi=110)
    FigureCanvasAgg(figure)
    for axis, segment in zip(
        figure.subplots(1, 3), ("tuning", "evaluation", "all"), strict=True
    ):
        for camera in (*inputs.camera_ids, "pooled"):
            curve = candidates[segment][camera]["all"]
            axis.plot(
                curve["k"],
                curve["recovered_rate"],
                marker=".",
                label=f"{camera} (n={curve['failed_argmax_count']})",
            )
        axis.set(
            title=segment,
            xlabel="Top-K",
            ylabel="Fraction of >20px argmax failures recovered",
            ylim=(-0.02, 1.02),
            xticks=range(1, 9),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    path = directory / "candidate-recovery.png"
    figure.savefig(path)
    figure.clear()
    output.append(path)

    figure = Figure(figsize=(13, 6), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for ax, subset in zip(axes, ("all_peaks", "within20_px"), strict=True):
        for i, camera in enumerate(inputs.camera_ids):
            for j, name in enumerate("xy"):
                row = decode["evaluation"][camera]["all"]["regression"][subset][name]
                if row["status"] != "measured":
                    continue
                index = i * 2 + j
                slope = row["slope_px_per_extent"]
                ci = row["coefficient_ci95"]
                ax.plot(index, slope, "o", color="#bb4f32")
                if ci is not None:
                    ax.plot([index, index], ci[1], color="#bb4f32")
                ax.plot(
                    [index - 0.12, index + 0.12],
                    [row["hypothesized_slope_magnitude"]] * 2,
                    color="#405fbb",
                )
                ax.plot(
                    [index - 0.12, index + 0.12],
                    [-row["hypothesized_slope_magnitude"]] * 2,
                    color="#405fbb",
                )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(
            xticks=range(6),
            xticklabels=[f"{c}/{n}" for c in inputs.camera_ids for n in "xy"],
            ylabel="Signed slope (px per normalized extent)",
            title=f"Evaluation / {subset}",
        )
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    figure.suptitle(
        "OLS + 95% source-time block bootstrap; blue = +/-0.375cell edge hypothesis\nWithin20px conditions on correct peaks; one clip cannot identify a causal resize bias"
    )
    figure.tight_layout()
    path = directory / "regression.png"
    figure.savefig(path)
    figure.clear()
    output.append(path)

    figure = Figure(figsize=(11, 4), dpi=110)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for offset, segment in zip(
        (-0.24, 0, 0.24), ("tuning", "evaluation", "all"), strict=True
    ):
        data = smoothing[segment]["pooled"]["all"]
        values = [
            data[event]["mean_reduction_px"] for event in ("bounce", "hit", "flight")
        ]
        axis.bar(np.arange(3) + offset, values, 0.24, label=segment)
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set(
        xticks=range(3),
        xticklabels=["Bounce (+/-3f)", "Hit (+/-3f)", "Flight (outside event union)"],
        ylabel="Raw error - LOESS error (mean px)",
        title="Matched accepted 2D observations; positive = smoothing helps",
    )
    axis.legend()
    figure.tight_layout()
    path = directory / "smoothing.png"
    figure.savefig(path)
    figure.clear()
    output.append(path)
    return output


def extend_report(path, figures, budget, timing):
    banner = """<h1>Scoring-only error decomposition</h1><p class="status">
Annotation, time-offset and candidate oracles below are diagnostics that use labels.
They are not deployable reconstruction methods. Clock parameters use all-camera tuning
labels only; this is a diagnostic calibration fit, not a blind LOVO estimate of clocks.
Candidate oracles select nearest labelled peaks in retained views only.
All LOVO residuals score the original held-out camera annotations at its own frame time.
Counterfactual differences are non-additive; use paired support and coverage.</p>"""
    extra = "<h2>Diagnostic figures</h2>"
    for figure in figures:
        encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
        extra += f'<h3>{html.escape(figure.stem)}</h3><img alt="{html.escape(figure.stem)}" src="data:image/png;base64,{encoded}">'
    extra += f"<h2>Paired error budget</h2><pre>{html.escape(json.dumps(budget, indent=2))}</pre>"
    extra += f"<h2>Tuning-only relative clock fit</h2><pre>{html.escape(json.dumps(timing, indent=2))}</pre>"
    page = path.read_text()
    page = page.replace("<h1>Ball 3D:", banner + "<h1>Ball 3D:")
    page = page.replace(
        "Each LOVO run sees only two cameras.",
        "Each inference LOVO run sees only two cameras; diagnostic clock fitting uses all-camera tuning labels as stated above.",
    )
    page = page.replace(
        "Coloured circles and court traces are method predictions.",
        "Coloured circles and court traces include explicitly named scoring-only oracles.",
    )
    path.write_text(page.replace("</html>", extra + "</html>"))
