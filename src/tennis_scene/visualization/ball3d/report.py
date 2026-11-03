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
from src.tennis_scene.evaluation.ball3d.contracts import MethodInput
from src.tennis_scene.evaluation.ball3d.runner import MethodRun
from src.tennis_scene.evaluation.ball3d.scoring import (
    error_samples,
    motion_samples,
    project_result,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.video.reader import RandomAccessVideoReader

from .rendering import method_colors


def plot_comparison(runs: Mapping[str, Sequence[MethodRun]], inputs: MethodInput, output: Path,
                    annotations: Ball2DAnnotations | None = None, *, split_frame: int = 505) -> None:
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
        axes[1, 0].plot(inputs.frame_indices, step, label=name, color=color, linewidth=.7)
        axes[2, 0].hist(motion["step_m"], bins=35, histtype="step", label=name, color=color)
        if annotations is not None:
            errors = []
            for fold in method_runs:
                if fold.held_out_camera is not None:
                    samples = error_samples(fold, inputs, annotations)
                    axes[1, 1].scatter(samples["frame_index"], samples["error_px"], s=3, alpha=.5, color=color)
                    errors.extend(samples["error_px"].tolist())
            axes[2, 1].hist(errors, bins=35, histtype="step", label=name, color=color)
        else:
            acceleration = np.full(inputs.num_frames, np.nan)
            acceleration[motion["acceleration_frames"]] = motion["acceleration_mps2"]
            axes[2, 1].plot(inputs.frame_indices, acceleration, label=name, color=color, linewidth=.7)
    axes[0, 0].set(xlabel="Court X (m)", ylabel="Court Y (m)", title="Top view: explicit 3D gaps")
    axes[0, 1].set(xlabel="Court Y (m)", ylabel="Court Z (m)", title="Side view: explicit 3D gaps")
    axes[1, 0].set(xlabel="Source frame", ylabel="Adjacent step (m)", title="Motion diagnostic (not accuracy)")
    axes[1, 0].axvline(split_frame, linestyle="--", color="gray", linewidth=.7)
    if annotations is not None:
        axes[1, 1].set(xlabel="Source frame", ylabel="Held-out error (px)", title="LOVO: directly observed labels only")
        axes[2, 1].set(xlabel="Held-out error (px)", ylabel="Count", title="LOVO residual distribution")
    else:
        axes[1, 1].text(.5, .5, "GT unavailable\nLOVO accuracy / events / oracle not measured",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_axis_off()
        axes[2, 1].set(xlabel="Source frame", ylabel="Acceleration (m/s²)", title="Motion diagnostic (not accuracy)")
    axes[2, 0].set(xlabel="Adjacent step (m)", ylabel="Count", title="Motion distribution (not accuracy)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output)
    figure.clear()


def render_extremes(runs: Mapping[str, Sequence[MethodRun]], inputs: MethodInput,
                    videos: Mapping[str, Path], output: Path, annotations: Ball2DAnnotations | None = None,
                    *, count: int = 6) -> dict[str, Any]:
    full = {name: next(r for r in values if r.held_out_camera is None) for name, values in runs.items()}
    ranking: dict[int, float] = {}
    basis = "largest adjacent 3D movement; not accuracy errors"
    if annotations is not None:
        basis = "largest observed-label reprojection error in full-view results"
        for run in full.values():
            samples = error_samples(run, inputs, annotations)
            for frame, value in zip(samples["frame_index"], samples["error_px"], strict=True):
                ranking[int(frame)] = max(ranking.get(int(frame), 0.), float(value))
    else:
        for run in full.values():
            motion = motion_samples(run.result, inputs.fps)
            for frame, value in zip(motion["step_frames"], motion["step_m"], strict=True):
                ranking[int(frame)] = max(ranking.get(int(frame), 0.), float(value))
    selected = sorted(ranking, key=lambda frame: (-ranking[frame], frame))[:count]
    if not selected:
        return {"selection_basis": basis, "frames": [], "status": "no_valid_samples_for_crops"}
    readers = {camera: RandomAccessVideoReader(path) for camera, path in videos.items()}
    colors = method_colors(tuple(full))
    projections = {(name, c): project_result(inputs, run.result, c) for name, run in full.items() for c in inputs.camera_ids}
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
                        cv2.circle(image, tuple(np.rint(center).astype(int)), 5, colors[name], 2, cv2.LINE_AA)
                if annotations is not None and annotations.observed[i, frame]:
                    center, center_source = annotations.uv_px[i, frame], "observed_annotation"
                    cv2.drawMarker(image, tuple(np.rint(center).astype(int)), (60, 255, 60), cv2.MARKER_CROSS, 14, 2)
                width, height = min(160, image.shape[1]), min(120, image.shape[0])
                x = int(np.clip(round(float(center[0])) - width // 2, 0, image.shape[1] - width))
                y = int(np.clip(round(float(center[1])) - height // 2, 0, image.shape[0] - height))
                crop = cv2.resize(image[y:y + height, x:x + width], (320, 240), interpolation=cv2.INTER_NEAREST)
                cv2.putText(crop, f"{camera} / frame {frame}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255, 255, 255), 1, cv2.LINE_AA)
                canvas[row * 240:(row + 1) * 240, i * 320:(i + 1) * 320] = crop
                records.append({"camera_id": camera, "frame_index": frame, "crop_xywh": [x, y, width, height],
                                "center_source": center_source, "ranking_value": ranking[frame]})
    finally:
        for reader in readers.values():
            reader.close()
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("Failed to save extreme-frame crops")
    return {"selection_basis": basis, "frames": selected, "crops": records, "status": "generated"}


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"src", "href"} and value is not None and not value.startswith(("data:", "#")):
                self.paths.append(value)


def broken_html_references(path: Path) -> list[str]:
    references = _References()
    references.feed(path.read_text())
    return [ref for ref in references.paths if not (path.parent / ref).is_file()]


def write_report(directory: Path, scores: dict[str, Any], *, video: Path, figure: Path,
                  crops: Path | None, verification: dict[str, Any]) -> Path:
    """Embed media and CSS; only optional download links refer to sibling files."""
    def embed(path: Path, mime: str) -> str:
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def number(value: Any) -> str:
        return "unavailable" if value is None else f"{value:.4g}"

    rows = []
    for name, method in scores["methods"].items():
        for segment in ("tuning", "evaluation", "all"):
            main = method["lovo_reprojection_px"][segment]
            validity = method["folds"]["all_views"][segment]["valid_3d_rate"]
            values = [html.escape(name), segment, number(None if main is None else main["median"]),
                      number(None if main is None else main["p90"]),
                      number(None if main is None else main["above_threshold_rate"]), number(validity)]
            rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    status = "Direct-video labels supplied; see provenance and coverage below." if scores["annotations"] else (
        "GT unavailable. Accuracy, observed-label LOVO, events and annotation oracle have not been measured. "
        "The projections and motion diagnostics below are predictions, not ground truth.")
    split = scores["config"]["split_frame"]
    stop = next(iter(scores["methods"].values()))["folds"]["all_views"]["all"]["frames"][1]
    threshold = scores["config"]["gross_reprojection_px"]
    crop_html = "" if crops is None else f'<h2>Selected source crops</h2><img alt="Source crops" src="{embed(crops, "image/png")}">'
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ball 3D comparison</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:2em auto;padding:0 1em;background:#111820;color:#e8edf1}}
img,video{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5em;border:1px solid #536170}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#95d0ff}}.status{{padding:1em;background:#293846}}</style>
<h1>Ball 3D: fixed-input CPU comparison</h1><p class="status">{html.escape(status)}</p>
<p>Tuning: [0,{split}). Primary evaluation: [{split},{stop}). Whole clip is supplementary. Missing labels/3D are never filled for scoring.
Green crosses are reviewed 2D labels; coloured circles and court traces are method predictions. Each LOVO run sees only two cameras.</p>
<table><thead><tr><th>Method</th><th>Segment</th><th>LOVO median px</th><th>LOVO p90 px</th><th>LOVO &gt;{threshold:g}px rate</th><th>Valid 3D rate</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Comparison video</h2><video controls preload="metadata" src="{embed(video, 'video/mp4')}"></video>
<h2>Trajectories and diagnostics</h2><img alt="Comparison plots" src="{embed(figure, 'image/png')}">{crop_html}
<h2>Provenance and verification</h2><pre>{html.escape(json.dumps(verification, ensure_ascii=False, indent=2))}</pre>
<details><summary>Complete metrics, thresholds and label coverage</summary><pre>{html.escape(json.dumps(scores, ensure_ascii=False, indent=2))}</pre></details>
<p><a href="scores.json">Metrics JSON</a> · <a href="comparison.mp4">Video download</a></p></html>'''
    write_json_atomic(directory / "scores.json", scores)
    path = directory / "index.html"
    path.write_text(report)
    if broken_html_references(path):
        raise ValueError("Generated comparison HTML has broken references")
    return path
