"""Portable comparison figures, extreme-frame crops and a self-contained HTML."""

from __future__ import annotations

import base64
import html
import json
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from src.tennis_scene.evaluation.ball3d.annotations import Ball2DAnnotations
from src.tennis_scene.evaluation.ball3d.bundle import FixedInputBundle
from src.tennis_scene.evaluation.ball3d.runner import MethodRun
from src.tennis_scene.evaluation.ball3d.scoring import (
    error_samples,
    motion_samples,
    project_result,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.video.reader import RandomAccessVideoReader

from .rendering import method_colors
from .tables import comparison_tables


def plot_comparison(
    runs: Mapping[str, Sequence[MethodRun]],
    bundle: FixedInputBundle,
    output: Path,
    annotations: Ball2DAnnotations | None = None,
    *,
    split_frame: int = 505,
) -> None:
    inputs = bundle.inputs
    figure = Figure(figsize=(12, 11), dpi=120)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    colors = method_colors(tuple(runs))
    for name, method_runs in runs.items():
        run = next(r for r in method_runs if r.held_out_camera is None)
        result = run.result
        color = tuple(c / 255 for c in reversed(colors[name]))
        xyz = np.where(result.valid[:, None], result.positions, np.nan)
        axes[0, 0].plot(xyz[:, 0], xyz[:, 1], label=name, color=color, linewidth=1)
        axes[0, 1].plot(xyz[:, 1], xyz[:, 2], label=name, color=color, linewidth=1)
        motion = motion_samples(result, inputs.fps)
        step = np.full(inputs.num_frames, np.nan)
        step[motion["step_frames"]] = motion["step_m"]
        axes[1, 0].plot(
            inputs.frame_indices, step, label=name, color=color, linewidth=0.7
        )
        axes[2, 0].hist(
            motion["step_m"], bins=35, histtype="step", label=name, color=color
        )
        if annotations is not None:
            errors = []
            for fold in method_runs:
                if fold.held_out_camera is not None:
                    samples = error_samples(fold, bundle, annotations)
                    axes[1, 1].scatter(
                        samples["frame_index"],
                        samples["error_px"],
                        s=3,
                        alpha=0.5,
                        color=color,
                    )
                    errors.extend(samples["error_px"].tolist())
            axes[2, 1].hist(errors, bins=35, histtype="step", label=name, color=color)
        else:
            acceleration = np.full(inputs.num_frames, np.nan)
            acceleration[motion["acceleration_frames"]] = motion["acceleration_mps2"]
            axes[2, 1].plot(
                inputs.frame_indices,
                acceleration,
                label=name,
                color=color,
                linewidth=0.7,
            )
    axes[0, 0].set(
        xlabel="Court X (m)", ylabel="Court Y (m)", title="Top view: explicit 3D gaps"
    )
    axes[0, 1].set(
        xlabel="Court Y (m)", ylabel="Court Z (m)", title="Side view: explicit 3D gaps"
    )
    axes[1, 0].set(
        xlabel="Source frame",
        ylabel="Adjacent step (m)",
        title="Motion diagnostic (not accuracy)",
    )
    axes[1, 0].axvline(split_frame, linestyle="--", color="gray", linewidth=0.7)
    if annotations is not None:
        axes[1, 1].set(
            xlabel="Source frame",
            ylabel="Held-out error (px)",
            title="LOVO: observed + approved estimated references",
        )
        axes[1, 1].axvline(split_frame, linestyle="--", color="gray", linewidth=0.7)
        axes[2, 1].set(
            xlabel="Held-out error (px)",
            ylabel="Count",
            title="LOVO residual distribution",
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "GT unavailable\nLOVO accuracy / events / oracle not measured",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_axis_off()
        axes[2, 1].set(
            xlabel="Source frame",
            ylabel="Acceleration (m/s²)",
            title="Motion diagnostic (not accuracy)",
        )
    axes[2, 0].set(
        xlabel="Adjacent step (m)",
        ylabel="Count",
        title="Motion distribution (not accuracy)",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output)
    figure.clear()


def plot_metrics_table(scores: dict[str, Any], output: Path) -> None:
    """Export the same split-wise comparison as a standalone figure for N methods."""
    tables = comparison_tables(scores)
    count = len(tables[0].rows)
    figure = Figure(figsize=(18, 3 + 0.85 * count), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1)
    for axis, spec in zip(axes, tables, strict=True):
        axis.set_axis_off()
        table = axis.table(
            cellText=spec.rows,
            colLabels=[
                header.replace(" ", "\n", 1)
                if header.startswith("All-view")
                else header
                for header in spec.headers
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.7)
        table.auto_set_column_width([0, 1])
        axis.set_title(spec.title, pad=22)
        axis.text(
            0.0, -0.05, spec.note, transform=axis.transAxes, fontsize=9, wrap=True
        )
    figure.tight_layout(pad=3)
    figure.savefig(output)
    figure.clear()


def render_extremes(
    runs: Mapping[str, Sequence[MethodRun]],
    bundle: FixedInputBundle,
    videos: Mapping[str, Path],
    output: Path,
    annotations: Ball2DAnnotations | None = None,
    *,
    count: int = 6,
) -> dict[str, Any]:
    inputs = bundle.inputs
    full = {
        name: next(r for r in values if r.held_out_camera is None)
        for name, values in runs.items()
    }
    ranking: dict[int, float] = {}
    basis = "largest adjacent 3D movement; not accuracy errors"
    if annotations is not None:
        basis = "largest held-out user-reference reprojection error across LOVO folds"
        for method_runs in runs.values():
            for run in method_runs:
                if run.held_out_camera is None:
                    continue
                samples = error_samples(run, bundle, annotations)
                for frame, value in zip(
                    samples["frame_index"], samples["error_px"], strict=True
                ):
                    ranking[int(frame)] = max(
                        ranking.get(int(frame), 0.0), float(value)
                    )
    else:
        for run in full.values():
            motion = motion_samples(run.result, inputs.fps)
            for frame, value in zip(
                motion["step_frames"], motion["step_m"], strict=True
            ):
                ranking[int(frame)] = max(ranking.get(int(frame), 0.0), float(value))
    selected = sorted(ranking, key=lambda frame: (-ranking[frame], frame))[:count]
    if not selected:
        return {
            "selection_basis": basis,
            "frames": [],
            "status": "no_valid_samples_for_crops",
        }
    readers = {camera: RandomAccessVideoReader(path) for camera, path in videos.items()}
    colors = method_colors(tuple(full))
    projections = {
        (name, c): project_result(inputs, run.result, c)
        for name, run in full.items()
        for c in inputs.camera_ids
    }
    canvas: NDArray[np.uint8] = np.full((len(selected) * 240, 3 * 320, 3), 23, np.uint8)
    records = []
    try:
        for row, frame in enumerate(selected):
            for i, camera in enumerate(inputs.camera_ids):
                image = readers[camera].read(frame)
                center = np.asarray(inputs.source_size_wh, np.float64) / 2
                center_source = "image_centre_when_no_projection"
                for name in full:
                    uv, valid = projections[name, camera]
                    if valid[frame]:
                        center, center_source = uv[frame], "method_projection"
                        cv2.circle(
                            image,
                            tuple(np.rint(center).astype(int)),
                            5,
                            colors[name],
                            2,
                            cv2.LINE_AA,
                        )
                if annotations is not None and annotations.valid_for_scoring[i, frame]:
                    observed = annotations.observed[i, frame]
                    center, center_source = (
                        annotations.uv_px[i, frame],
                        str(annotations.reference_kind[i, frame]) + "_reference",
                    )
                    cv2.drawMarker(
                        image,
                        tuple(np.rint(center).astype(int)),
                        (60, 255, 60) if observed else (255, 255, 255),
                        cv2.MARKER_CROSS if observed else cv2.MARKER_DIAMOND,
                        14,
                        2,
                    )
                for name, method_runs in runs.items():
                    fold = next(
                        run for run in method_runs if run.held_out_camera == camera
                    )
                    uv, valid = project_result(inputs, fold.result, camera)
                    if valid[frame]:
                        cv2.drawMarker(
                            image,
                            tuple(np.rint(uv[frame]).astype(int)),
                            colors[name],
                            cv2.MARKER_TILTED_CROSS,
                            12,
                            2,
                        )
                width, height = min(160, image.shape[1]), min(120, image.shape[0])
                x = int(
                    np.clip(
                        round(float(center[0])) - width // 2, 0, image.shape[1] - width
                    )
                )
                y = int(
                    np.clip(
                        round(float(center[1])) - height // 2,
                        0,
                        image.shape[0] - height,
                    )
                )
                crop = cv2.resize(
                    image[y : y + height, x : x + width],
                    (320, 240),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.putText(
                    crop,
                    f"{camera} / f{frame} | max {ranking[frame]:.1f}{'px' if annotations is not None else 'm'}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                canvas[row * 240 : (row + 1) * 240, i * 320 : (i + 1) * 320] = crop
                records.append(
                    {
                        "camera_id": camera,
                        "frame_index": frame,
                        "crop_xywh": [x, y, width, height],
                        "center_source": center_source,
                        "ranking_value": ranking[frame],
                    }
                )
    finally:
        for reader in readers.values():
            reader.close()
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("Failed to save extreme-frame crops")
    return {
        "selection_basis": basis,
        "frames": selected,
        "crops": records,
        "status": "generated",
    }


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if (
                key in {"src", "href"}
                and value is not None
                and not value.startswith(("data:", "#"))
            ):
                self.paths.append(value)


def broken_html_references(path: Path) -> list[str]:
    references = _References()
    references.feed(path.read_text())
    return [ref for ref in references.paths if not (path.parent / ref).is_file()]


def write_report(
    directory: Path,
    scores: dict[str, Any],
    *,
    video: Path,
    figure: Path,
    crops: Path | None,
    verification: dict[str, Any],
    table_figure: Path | None = None,
    supplementary_figures: Mapping[str, Path] | None = None,
) -> Path:
    """Embed media and CSS; only optional download links refer to sibling files."""

    def embed(path: Path, mime: str) -> str:
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode(
            "ascii"
        )

    def number(value: Any) -> str:
        return "unavailable" if value is None else f"{value:.4g}"

    tables = comparison_tables(scores)
    supplementary = "".join(
        f'<h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="{embed(path, "image/png")}">'
        for title, path in (supplementary_figures or {}).items()
    )
    breakdown_rows = []
    for name, method in scores["methods"].items():
        for segment in ("tuning", "evaluation", "all"):
            for kind in ("observed", "estimated"):
                summaries = method["lovo_reprojection_by_reference_kind"][segment]
                summary = None if summaries is None else summaries[kind]
                values = [
                    html.escape(name),
                    segment,
                    kind,
                    number(None if summary is None else summary["median"]),
                    number(None if summary is None else summary["p90"]),
                    number(
                        None if summary is None else summary["above_threshold_rate"]
                    ),
                    number(None if summary is None else summary["count"]),
                    number(None if summary is None else summary["label_count"]),
                    number(
                        None if summary is None else summary["scored_label_coverage"]
                    ),
                ]
                breakdown_rows.append(
                    "<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>"
                )
    status = (
        "User reference supplied: directly observed and user-approved estimated coordinates are both scored; see separate breakdowns."
        if scores["annotations"]
        else (
            "GT unavailable. Accuracy, observed-label LOVO, events and annotation oracle have not been measured. "
            "The projections and motion diagnostics below are predictions, not ground truth."
        )
    )
    split = scores["config"]["split_frame"]
    stop = next(iter(scores["methods"].values()))["folds"]["all_views"]["all"][
        "frames"
    ][1]
    threshold = scores["config"]["gross_reprojection_px"]
    crop_html = (
        ""
        if crops is None
        else f'<h2>Selected source crops</h2><img alt="Source crops" src="{embed(crops, "image/png")}">'
    )
    table_html = (
        ""
        if table_figure is None
        else f'<h2>Split-wise metric table</h2><img alt="Metric table" src="{embed(table_figure, "image/png")}">'
    )
    table_sections = "".join(
        f'<section id="{spec.key}"><h2>{html.escape(spec.title)}</h2><p>{html.escape(spec.note)}</p><table><thead><tr>'
        + "".join(f"<th>{html.escape(header)}</th>" for header in spec.headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in row)
            + "</tr>"
            for row in spec.rows
        )
        + "</tbody></table></section>"
        for spec in tables
    )
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ball 3D comparison</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:2em auto;padding:0 1em;background:#111820;color:#e8edf1}}
img,video{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5em;border:1px solid #536170}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#95d0ff}}.status{{padding:1em;background:#293846}}</style>
<h1>Ball 3D: fixed-input CPU comparison</h1><p class="status">{html.escape(status)}</p>
<p>Tuning: [0,{split}). Primary evaluation: [{split},{stop}). Whole clip is supplementary. Missing labels/3D are never filled for scoring.
Green crosses are directly observed references; white diamonds are approved estimates. Coloured circles and court traces are method predictions.
In crops, coloured crosses are LOVO predictions and rings are all-view predictions. Each LOVO run sees only two cameras.
Camera-specific event rows retain their notes and one-frame differences. Event errors use the union of windows; rows are not counted as physical events.
The annotation oracle uses the same approximate cameras and is a geometric reference, not a guaranteed mathematical lower bound.</p>
{table_sections}
<details><summary>LOVO reference-kind breakdown and coverage</summary><table><thead><tr><th>Method</th><th>Segment</th><th>Reference kind</th>
<th>Median px</th><th>p90 px</th><th>&gt;{threshold:g}px rate</th><th>Scored</th><th>Eligible</th><th>Coverage</th></tr></thead>
<tbody>{"".join(breakdown_rows)}</tbody></table></details>{table_html}
<h2>Comparison video</h2><video controls preload="metadata" src="{embed(video, "video/mp4")}"></video>
<h2>Trajectories and diagnostics</h2><img alt="Comparison plots" src="{embed(figure, "image/png")}">{crop_html}
{supplementary}<h2>Provenance and verification</h2><pre>{html.escape(json.dumps(verification, ensure_ascii=False, indent=2))}</pre>
<details><summary>Complete metrics, thresholds and label coverage</summary><pre>{html.escape(json.dumps(scores, ensure_ascii=False, indent=2))}</pre></details>
<p>Metrics are embedded above; sibling files provide optional downloads.</p>
<p><a href="scores.json">Metrics JSON</a> · <a href="comparison.mp4">Video download</a></p></html>'''
    write_json_atomic(directory / "scores.json", scores)
    path = directory / "index.html"
    path.write_text(report)
    if broken_html_references(path):
        raise ValueError("Generated comparison HTML has broken references")
    return path
