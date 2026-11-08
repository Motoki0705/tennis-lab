"""Clip comparison diagnostics, after blind method execution."""

from __future__ import annotations

import csv

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def diagnostic_summary(runs, fps):
    rows = []
    for name, values in runs.items():
        if not name.startswith("segment_physics_"):
            continue
        for run in values:
            d = run.result.diagnostics
            stderr = d["offset_standard_error_seconds"]
            rows.append(
                {
                    "method": name,
                    "fold": run.key,
                    "camera_ids": list(run.result.camera_ids),
                    "offset_frames": (run.result.time_offsets_seconds * fps).tolist(),
                    "offset_standard_error_frames": None
                    if stderr is None
                    else (np.asarray(stderr) * fps).tolist(),
                    "anchor": d["anchor_camera"],
                    "status": d["status"],
                    "segments": d["segments"],
                    "events": d["events"],
                    **{
                        k: d.get(k)
                        for k in (
                            "drag_per_m",
                            "drag_standard_error_per_m",
                            "nfev",
                            "information_rank",
                            "parameter_count",
                            "uncertainty_at_bound",
                            "uncertainty_unavailable_reason",
                        )
                    },
                    "valid_3d_frames": int(run.result.valid.sum()),
                    "runtime_seconds": run.elapsed_seconds,
                }
            )
    return rows


def physical_defect(positions, valid, segment_ids, fps, drag, gravity):
    selected = (
        valid[:-2]
        & valid[1:-1]
        & valid[2:]
        & (segment_ids[:-2] == segment_ids[2:])
        & (segment_ids[1:-1] >= 0)
    )
    frames = np.flatnonzero(selected) + 1
    velocity = (positions[frames + 1].astype(float) - positions[frames - 1]) * fps / 2
    acceleration = (
        positions[frames + 1].astype(float)
        - 2 * positions[frames]
        + positions[frames - 1]
    ) * fps**2
    return frames, np.linalg.norm(
        acceleration
        + [0, 0, gravity]
        + drag * np.linalg.norm(velocity, axis=1)[:, None] * velocity,
        axis=1,
    )


def plot_diagnostics(runs, fps, path):
    physics = {
        name: values[0]
        for name, values in runs.items()
        if name.startswith("segment_physics_")
    }
    baseline = runs["pipeline_baseline"][0].result
    figure = Figure(figsize=(14, 10), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    summary = diagnostic_summary(runs, fps)
    for col, (name, run) in enumerate(physics.items()):
        result, d = run.result, run.result.diagnostics
        xyz = np.where(result.valid[:, None], result.positions, np.nan)
        for index, segment in enumerate(d["segments"]):
            part = xyz[segment["start_frame"] : segment["stop_frame"]]
            axes[0, col].plot(
                part[:, 1],
                part[:, 2],
                linewidth=0.7,
                color="C0",
                label=name if index == 0 else None,
            )
        if not result.valid.any():
            axes[0, col].text(
                0.5,
                0.5,
                f"{d['status']}\n{d['nfev']} evaluations; no valid output",
                ha="center",
                va="center",
                transform=axes[0, col].transAxes,
            )
        for event in d["events"]:
            f = event["frame"]
            if result.valid[f]:
                axes[0, col].scatter(*result.positions[f, [1, 2]], marker="x", s=25)
                axes[0, col].annotate(
                    f"{event['kind']} {f}", result.positions[f, [1, 2]], fontsize=6
                )
        for segment in d["segments"]:
            f = segment["start_frame"]
            if result.valid[f]:
                axes[0, col].scatter(
                    *result.positions[f, [1, 2]], marker="o", s=8, color="gray"
                )
        axes[0, col].set(
            xlabel="Court Y (m)",
            ylabel="Court Z (m)",
            title="Segments / predicted events",
        )
        if d.get("drag_per_m") is not None:
            for label, positions, valid in (
                ("baseline", baseline.positions, baseline.valid),
                (name, result.positions, result.valid),
            ):
                frame, defect = physical_defect(
                    positions,
                    valid,
                    np.asarray(d["frame_segment"]),
                    fps,
                    d["drag_per_m"],
                    d["config"]["gravity_mps2"],
                )
                if not len(frame):
                    continue
                axes[1, col].semilogy(
                    frame, np.maximum(defect, 1e-4), linewidth=0.6, label=label
                )
            for event in d["events"]:
                axes[1, col].axvline(event["frame"], color="gray", alpha=0.25)
        axes[1, col].set(
            xlabel="Source frame",
            ylabel="ODE defect (m/s²)",
            title="Finite-difference a + g + k |v| v (within-flight stencils)",
        )
        if d["status"] != "ok":
            axes[1, col].text(
                0.5,
                0.9,
                "No valid physical fit; baseline shown using failed-iterate drag",
                ha="center",
                fontsize=8,
                transform=axes[1, col].transAxes,
            )
        labels, offsets, errors = [], [], []
        for row in summary:
            if row["method"] != name:
                continue
            for i, camera in enumerate(row["camera_ids"]):
                labels.append(f"{row['fold']} / {camera}")
                offsets.append(row["offset_frames"][i])
                errors.append(
                    np.nan
                    if row["offset_standard_error_frames"] is None
                    else row["offset_standard_error_frames"][i]
                )
        yy = np.arange(len(labels))
        axes[2, col].scatter(offsets, yy, s=20)
        finite = np.isfinite(errors)
        axes[2, col].errorbar(
            np.asarray(offsets)[finite],
            yy[finite],
            xerr=np.asarray(errors)[finite],
            fmt="none",
            capsize=3,
        )
        axes[2, col].axvline(0, color="gray", linewidth=0.6)
        axes[2, col].set(
            yticks=yy,
            yticklabels=labels,
            xlabel="Relative offset (source frames)",
            title="Offsets ± local standard error; missing bars = unavailable"
            if d["status"] == "ok"
            else "Final FAILED iterates; uncertainty unavailable",
        )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for axis in axes[:2].flat:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    figure.clear()


def export_tables(scores, directory):
    rows = []
    for name, method in scores["methods"].items():
        for split in ("evaluation", "tuning", "all"):
            for fold, summary in [
                ("lovo_pool", method["lovo_reprojection_px"][split]),
                ("common_lovo", method["common_lovo_reprojection_px"][split]),
                *(
                    (key, values[split]["reprojection_by_reference_kind"]["all"])
                    for key, values in method["folds"].items()
                ),
            ]:
                rows.append(
                    {
                        "method": name,
                        "split": split,
                        "fold": fold,
                        **{
                            k: summary[k]
                            for k in (
                                "median",
                                "p90",
                                "above_threshold_rate",
                                "count",
                                "label_count",
                                "scored_label_coverage",
                            )
                        },
                    }
                )
    with (directory / "fold-metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "値は median / p90 px / >20px率 ; 採点数/対象数 (coverage)。",
        "",
        "| 区間 | 方式 | LOVO pool | cam0除外 | cam1除外 | cam2除外 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("evaluation", "tuning", "all"):
        for name in scores["methods"]:
            values = []
            for fold in ("lovo_pool", "without_cam0", "without_cam1", "without_cam2"):
                row = next(
                    r
                    for r in rows
                    if r["method"] == name and r["split"] == split and r["fold"] == fold
                )
                if row["count"]:
                    values.append(
                        f"{row['median']:.3f}/{row['p90']:.3f}/{row['above_threshold_rate']:.2%}; {row['count']}/{row['label_count']} ({row['scored_label_coverage']:.2%})"
                    )
                else:
                    values.append(f"unavailable; 0/{row['label_count']} (0%)")
            text.append(f"| {split} | {name} | " + " | ".join(values) + " |")
    (directory / "metric-tables.md").write_text("\n".join(text) + "\n")
    return rows
