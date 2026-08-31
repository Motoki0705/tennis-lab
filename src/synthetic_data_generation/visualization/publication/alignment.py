"""Alignment progression and metric heatmap/court publication figures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import (
    ALIGNMENT_TRACE_SCHEMA,
    GROUND_PLANE_FRAME_SCHEMA,
    AlignmentEvidence,
    AlignmentResult,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    LINE_HEATMAP_DIRECTORY,
    AlignmentLineHeatmaps,
    aggregate_line_heatmaps,
    validate_line_heatmaps,
)
from src.synthetic_data_generation.alignment.validation import (
    GROUND_LINE_MAP_FILE,
    load_alignment_evidence,
    validate_alignment_outputs,
)
from src.synthetic_data_generation.visualization.publication.datasets import (
    write_deterministic_gif,
)
from src.utils.schema.court import (
    COURT_SKELETON,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

ALIGNMENT_AGREEMENT_METRIC_SCHEMA = "alignment_heatmap_court_agreement_v1"
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


@dataclass(frozen=True, slots=True)
class AlignmentPublicationData:
    """Validated alignment contracts and recomputed agreement evidence."""

    result: AlignmentResult
    evidence: AlignmentEvidence
    heatmaps: AlignmentLineHeatmaps
    court_segments_uv: tuple[NDArray[np.float64], ...]
    court_line_samples_uv: NDArray[np.float64]
    projected_evidence_uv: NDArray[np.float64]
    projected_probabilities: NDArray[np.float64]
    metrics: Mapping[str, object]


def load_alignment_publication_data(root: Path) -> AlignmentPublicationData:
    """Validate the complete owner and bind trace, plane, heatmaps, and final court."""
    result = validate_alignment_outputs(root)
    evidence = load_alignment_evidence(root / GROUND_LINE_MAP_FILE)
    heatmaps = validate_line_heatmaps(root / LINE_HEATMAP_DIRECTORY)
    if result.metric_adapter.to_dict() != evidence.metric_adapter.to_dict():
        raise ValueError("Alignment result/evidence metric adapters differ.")
    if heatmaps.bounds_uv != evidence.ground_plane_frame.bounds_uv_metres:
        raise ValueError(
            "Line heatmaps and persisted metric ground-plane bounds differ."
        )
    court_segments = _metric_court_segments_uv(result, evidence=evidence)
    court_line_samples = np.concatenate(
        [
            np.linspace(segment[0], segment[1], num=64, endpoint=True, dtype=np.float64)
            for segment in court_segments
        ]
    )
    projected_points: list[NDArray[np.float64]] = []
    projected_probabilities: list[NDArray[np.float64]] = []
    for view in heatmaps.views:
        if view.included_in_aggregate:
            projected_points.append(view.points_uv)
            projected_probabilities.append(
                view.projected_probabilities.astype(np.float64, copy=False)
            )
    if not projected_points or sum(len(item) for item in projected_points) == 0:
        raise ValueError("Alignment publication requires projected aggregate evidence.")
    points = np.concatenate(projected_points)
    probabilities = np.concatenate(projected_probabilities)
    metrics = compute_alignment_agreement_metrics(
        heatmaps,
        court_line_samples_uv=court_line_samples,
        projected_evidence_uv=points,
        court_segments_uv=court_segments,
        evidence=evidence,
        result=result,
    )
    return AlignmentPublicationData(
        result=result,
        evidence=evidence,
        heatmaps=heatmaps,
        court_segments_uv=court_segments,
        court_line_samples_uv=court_line_samples,
        projected_evidence_uv=points,
        projected_probabilities=probabilities,
        metrics=metrics,
    )


def compute_alignment_agreement_metrics(
    heatmaps: AlignmentLineHeatmaps,
    *,
    court_line_samples_uv: NDArray[np.float64],
    projected_evidence_uv: NDArray[np.float64],
    court_segments_uv: tuple[NDArray[np.float64], ...],
    evidence: AlignmentEvidence,
    result: AlignmentResult,
) -> Mapping[str, object]:
    """Compute documented raster probability and metric nearest-line agreement."""
    samples = np.asarray(court_line_samples_uv, dtype=np.float64)
    projected = np.asarray(projected_evidence_uv, dtype=np.float64)
    if (
        samples.ndim != 2
        or samples.shape[1:] != (2,)
        or len(samples) == 0
        or projected.ndim != 2
        or projected.shape[1:] != (2,)
        or len(projected) == 0
        or not np.isfinite(samples).all()
        or not np.isfinite(projected).all()
    ):
        raise ValueError("Agreement inputs must be non-empty finite (N, 2) arrays.")
    raster = aggregate_line_heatmaps(heatmaps).mean_probability
    u_min, _, v_min, _ = heatmaps.bounds_uv
    columns = np.rint((samples[:, 0] - u_min) / heatmaps.grid_spacing).astype(np.int64)
    rows = np.rint((samples[:, 1] - v_min) / heatmaps.grid_spacing).astype(np.int64)
    if (
        np.any(columns < 0)
        or np.any(columns >= raster.shape[1])
        or np.any(rows < 0)
        or np.any(rows >= raster.shape[0])
    ):
        raise ValueError("Accepted court lines exceed the persisted heatmap grid.")
    court_probability = raster[rows, columns].astype(np.float64)
    distances = _nearest_segment_distances(projected, court_segments_uv)
    local_points = _court_ground_points()
    plane_errors = tuple(
        np.abs(
            evidence.ground_plane_frame.signed_distances(
                court.scene_from_court.apply(local_points)
            )
        )
        for court in result.layout.courts
    )
    maximum_plane_error = max(float(np.max(value)) for value in plane_errors)
    if maximum_plane_error > 1.0e-6:
        raise ValueError(
            "Accepted metric court disagrees with the persisted ground plane."
        )
    return {
        "schema": ALIGNMENT_AGREEMENT_METRIC_SCHEMA,
        "court_line_sample_count": len(samples),
        "projected_evidence_point_count": len(projected),
        "court_line_mean_probability": float(np.mean(court_probability)),
        "court_line_probability_q50": float(np.quantile(court_probability, 0.5)),
        "court_line_coverage_fraction_at_0_5": float(np.mean(court_probability >= 0.5)),
        "projected_evidence_nearest_court_mean_metres": float(np.mean(distances)),
        "projected_evidence_nearest_court_q95_metres": float(
            np.quantile(distances, 0.95)
        ),
        "ground_plane_binding_max_abs_error_metres": maximum_plane_error,
    }


def render_alignment_progression_gif(
    data: AlignmentPublicationData,
    output: Path,
    *,
    size: tuple[int, int],
    duration_ms: int,
    line_width: float,
    font_size: int,
) -> tuple[Mapping[str, object], ...]:
    """Render exactly one frame for every persisted four-phase trace state."""
    trace = data.evidence.alignment_trace
    frames: list[NDArray[np.uint8]] = []
    mapping: list[Mapping[str, object]] = []
    court_points = _court_ground_points()[:, :2]
    segments = tuple(
        (first, second)
        for first, second in COURT_SKELETON
        if first < len(court_points) and second < len(court_points)
    )
    bounds = data.evidence.ground_plane_frame.bounds_uv_metres
    for step in trace.steps:
        figure = Figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        axis.set_facecolor("#F7F7F7")
        axis.scatter(
            data.projected_evidence_uv[:, 0],
            data.projected_evidence_uv[:, 1],
            c="#999999",
            s=3,
            alpha=0.20,
            linewidths=0,
            label="projected line evidence",
        )
        for candidate_index, candidate in enumerate(step.candidates):
            rotation = np.asarray(
                (
                    (
                        math.cos(candidate.orientation_radians),
                        -math.sin(candidate.orientation_radians),
                    ),
                    (
                        math.sin(candidate.orientation_radians),
                        math.cos(candidate.orientation_radians),
                    ),
                ),
                dtype=np.float64,
            )
            points = court_points @ rotation.T + np.asarray(candidate.center_uv_metres)
            color = _COLORS[candidate_index % len(_COLORS)]
            for first, second in segments:
                axis.plot(
                    points[[first, second], 0],
                    points[[first, second], 1],
                    color=color,
                    linewidth=line_width,
                )
            axis.scatter(
                [candidate.center_uv_metres[0]],
                [candidate.center_uv_metres[1]],
                color=color,
                s=25,
                label=(
                    f"{candidate.candidate_id}: score={candidate.template_score:.3f}, "
                    f"scale={candidate.nht_scene_units_per_metre:.4f} NHT/m"
                ),
            )
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("ground-plane u (m)", fontsize=font_size)
        axis.set_ylabel("ground-plane v (m)", fontsize=font_size)
        axis.set_title(
            f"Alignment phase {step.step_index + 1}/4 — {step.phase.value}\n"
            f"candidate score sum={step.score_sum:.4f}",
            fontsize=font_size + 2,
        )
        axis.grid(color="#DDDDDD", linewidth=0.6)
        axis.legend(loc="upper right", fontsize=max(6, font_size - 2), framealpha=0.95)
        figure.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.86)
        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        if frame.shape != (size[1], size[0], 3):
            raise ValueError("Alignment progression canvas dimensions changed.")
        frames.append(frame)
        mapping.append(
            {
                "step_index": step.step_index,
                "phase": step.phase.value,
                "score_sum": step.score_sum,
                "candidate_ids": [item.candidate_id for item in step.candidates],
                "candidate_scores": [item.template_score for item in step.candidates],
            }
        )
    write_deterministic_gif(tuple(frames), output, duration_ms=duration_ms)
    return tuple(mapping)


def render_alignment_heatmap_court_png(
    data: AlignmentPublicationData,
    output: Path,
    *,
    size: tuple[int, int],
    line_width: float,
    font_size: int,
) -> None:
    """Render probability, projected evidence, and metric court on common UV axes."""
    rasters = aggregate_line_heatmaps(data.heatmaps)
    bounds = data.heatmaps.bounds_uv
    figure = Figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(
        rasters.mean_probability,
        origin="lower",
        extent=(bounds[0], bounds[1], bounds[2], bounds[3]),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    positive = data.projected_probabilities > 0.0
    axis.scatter(
        data.projected_evidence_uv[positive, 0],
        data.projected_evidence_uv[positive, 1],
        c=data.projected_probabilities[positive],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        s=4,
        alpha=0.35,
        linewidths=0,
        label="projected detector evidence",
    )
    for index, segment in enumerate(data.court_segments_uv):
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            color="#00FFFF",
            linewidth=line_width,
            label="accepted metric court" if index == 0 else None,
        )
    metrics = data.metrics
    axis.text(
        0.015,
        0.02,
        (
            "court-line mean probability="
            f"{_metric_value(metrics, 'court_line_mean_probability'):.3f}\n"
            "projected evidence q95 distance="
            f"{_metric_value(metrics, 'projected_evidence_nearest_court_q95_metres'):.3f} m\n"
            "plane binding max error="
            f"{_metric_value(metrics, 'ground_plane_binding_max_abs_error_metres'):.2e} m"
        ),
        transform=axis.transAxes,
        fontsize=font_size,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.75, "pad": 6},
    )
    axis.set_xlabel("ground-plane u (m)", fontsize=font_size)
    axis.set_ylabel("ground-plane v (m)", fontsize=font_size)
    axis.set_title(
        "Line probability + projected evidence + accepted metric court",
        fontsize=font_size + 2,
    )
    axis.legend(loc="upper right", fontsize=max(6, font_size - 1))
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("weighted mean line probability", fontsize=font_size)
    figure.subplots_adjust(left=0.09, right=0.91, bottom=0.09, top=0.92)
    _save_canvas_png(canvas, output, size=size)


def _metric_court_segments_uv(
    result: AlignmentResult,
    *,
    evidence: AlignmentEvidence,
) -> tuple[NDArray[np.float64], ...]:
    local_points = _court_ground_points()
    segments: list[NDArray[np.float64]] = []
    for court in result.layout.courts:
        metric_points = court.scene_from_court.apply(local_points)
        uv = evidence.ground_plane_frame.to_uv(metric_points)
        for first, second in COURT_SKELETON:
            if first < len(uv) and second < len(uv):
                segments.append(np.asarray(uv[[first, second]], dtype=np.float64))
    if not segments:
        raise ValueError("Accepted layout provides no renderable court line segments.")
    return tuple(segments)


def _court_ground_points() -> NDArray[np.float64]:
    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].detach().cpu().numpy()
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (14, 3) or not np.isfinite(result).all():
        raise ValueError(
            "Canonical court geometry must contain fourteen ground points."
        )
    return result


def _nearest_segment_distances(
    points: NDArray[np.float64],
    segments: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.full(len(points), np.inf, dtype=np.float64)
    for segment in segments:
        start, stop = segment
        vector = stop - start
        denominator = float(vector @ vector)
        if denominator <= 0.0:
            raise ValueError("Court line segment has zero metric length.")
        projection = np.clip(((points - start) @ vector) / denominator, 0.0, 1.0)
        closest = start + projection[:, None] * vector
        result = np.minimum(result, np.linalg.norm(points - closest, axis=1))
    if not np.isfinite(result).all():
        raise ValueError("Court nearest-line distance could not be computed.")
    return result


def _metric_value(metrics: Mapping[str, object], name: str) -> float:
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Alignment metric {name!r} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Alignment metric {name!r} must be finite.")
    return result


def _save_canvas_png(
    canvas: FigureCanvasAgg,
    output: Path,
    *,
    size: tuple[int, int],
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Publication staging artifact already exists: {output}")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    if rgba.shape != (size[1], size[0], 4):
        raise ValueError("Publication PNG canvas dimensions changed.")
    from PIL import Image

    Image.fromarray(rgba[..., :3], mode="RGB").save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )


__all__ = [
    "ALIGNMENT_AGREEMENT_METRIC_SCHEMA",
    "ALIGNMENT_TRACE_SCHEMA",
    "AlignmentPublicationData",
    "GROUND_PLANE_FRAME_SCHEMA",
    "compute_alignment_agreement_metrics",
    "load_alignment_publication_data",
    "render_alignment_heatmap_court_png",
    "render_alignment_progression_gif",
]
