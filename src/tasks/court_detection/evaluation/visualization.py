"""Deterministic qualitative montages and summary figures.

Sample selection is a fixed quantile ladder over the primary model's normalized
keypoint error, plus every alignment failure, so the published figures cannot
be cherry-picked to the best cases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

matplotlib.use("Agg")

from src.tasks.court_detection.evaluation.alignment import (  # noqa: E402
    COURT_LINE_SEGMENTS,
    doubles_polygon_template,
    project_points,
    segment_polylines,
)
from src.tasks.court_detection.evaluation.contracts import (  # noqa: E402
    KEYPOINT_COUNT,
    LoadedSample,
    ModelPrediction,
)
from src.tasks.court_detection.evaluation.metrics import SampleEvaluation  # noqa: E402
from src.tasks.court_detection.evaluation.settings import (  # noqa: E402
    BenchmarkQualitySettings,
)

# OpenCV draws in BGR order, so these constants are written as the BGR triples
# that render as green / orange / blue in the saved PNG.
GT_COLOR = (80, 220, 80)
OURS_COLOR = (60, 170, 255)
TCD_COLOR = (255, 90, 60)
TEXT_COLOR = (245, 245, 245)
BACKGROUND = (18, 18, 18)
KEYPOINT_RADIUS = 5
PANEL_WIDTH = 640
LABEL_HEIGHT = 26
HEADER_HEIGHT = 30
TILE_GAP = 8

# Quantile ladder plus worst case; alignment failures are appended separately.
QUANTILE_LADDER: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
MODEL_ORDER = ("ours", "tcd")


@dataclass(frozen=True, slots=True)
class SelectedSample:
    sample_id: str
    domain: str
    scene_id: str
    primary_model: str
    median_error_px: float | None
    alignment_failed: bool
    selection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PanelGeometry:
    """Mapping between a source frame and a fixed-width rendered panel."""

    source_height: int
    source_width: int
    panel_height: int
    panel_width: int

    def scale(self, points: NDArray[np.floating]) -> NDArray[np.float64]:
        array = np.asarray(points, dtype=np.float64).copy()
        array[:, 0] *= self.panel_width / float(self.source_width)
        array[:, 1] *= self.panel_height / float(self.source_height)
        return array


def select_review_samples(
    evaluations: Sequence[SampleEvaluation],
    *,
    primary_model: str,
    count: int,
) -> tuple[SelectedSample, ...]:
    """Pick best/quartile/median/worst plus every alignment failure."""
    if count <= 0:
        raise ValueError("Review sample count must be positive.")
    primary = [item for item in evaluations if item.model == primary_model]
    if not primary:
        raise ValueError(f"No evaluations exist for primary model {primary_model!r}.")
    ranked = sorted(
        primary,
        key=lambda item: (
            float("inf") if item.median_error_px is None else item.median_error_px,
            item.ref.sample_id,
        ),
    )
    order = {item.ref.sample_id: index for index, item in enumerate(ranked)}
    chosen: dict[str, SelectedSample] = {}

    def remember(item: SampleEvaluation, reason: str) -> None:
        previous = chosen.get(item.ref.sample_id)
        reasons = (
            (reason,) if previous is None else (*previous.selection_reasons, reason)
        )
        median = item.median_error_px
        chosen[item.ref.sample_id] = SelectedSample(
            sample_id=item.ref.sample_id,
            domain=item.ref.domain,
            scene_id=item.ref.scene_id,
            primary_model=primary_model,
            median_error_px=median,
            alignment_failed=item.pred_homography is None,
            selection_reasons=tuple(sorted(set(reasons))),
        )

    for quantile in QUANTILE_LADDER:
        index = min(len(ranked) - 1, int(round(quantile * (len(ranked) - 1))))
        remember(ranked[index], f"quantile_{quantile:g}")
    for item in sorted(primary, key=lambda value: value.ref.sample_id):
        if item.pred_homography is None:
            remember(item, "alignment_failure")

    ordered = sorted(
        chosen.values(),
        key=lambda item: (not item.alignment_failed, order[item.sample_id]),
    )
    return tuple(ordered[:count])


def render_sample_row(
    sample: LoadedSample,
    predictions: Mapping[str, ModelPrediction],
    evaluations: Mapping[str, SampleEvaluation],
    *,
    quality: BenchmarkQualitySettings,
) -> NDArray[np.uint8]:
    """Render one review row: ground truth, each model, then a combined overlay."""
    # Keep every regulation line as its own polyline; a concatenated array would
    # draw spurious bridges between unrelated court lines.
    template_lines = segment_polylines(
        COURT_LINE_SEGMENTS,
        sample.template_xy,
        per_segment=quality.line_samples_per_segment,
    )
    polygon = doubles_polygon_template(sample.template_xy)
    panels: list[NDArray[np.uint8]] = []

    # The ground-truth homography is derived from the loaded sample, so every
    # model's evaluation of this sample carries the same one.
    reference = next(iter(evaluations.values()), None)
    gt_panel = _base_panel(sample)
    geometry = _geometry(sample, gt_panel)
    if reference is not None and reference.gt_homography is not None:
        _draw_court_lines(
            gt_panel, geometry, template_lines, reference.gt_homography, GT_COLOR
        )
        _draw_polygon(
            gt_panel,
            geometry,
            project_points(polygon, reference.gt_homography),
            GT_COLOR,
        )
    _draw_keypoints(
        gt_panel, geometry, sample.gt_keypoints_xy, sample.gt_visible, GT_COLOR
    )
    panels.append(
        _label(gt_panel, f"GT  visible={int(sample.gt_visible.sum())}/{KEYPOINT_COUNT}")
    )

    for model in MODEL_ORDER:
        evaluation = evaluations.get(model)
        prediction = predictions.get(model)
        if evaluation is None or prediction is None:
            continue
        color = OURS_COLOR if model == "ours" else TCD_COLOR
        panel = _base_panel(sample)
        geometry = _geometry(sample, panel)
        if evaluation.pred_homography is not None:
            _draw_court_lines(
                panel,
                geometry,
                template_lines,
                evaluation.pred_homography,
                color,
            )
            _draw_polygon(
                panel,
                geometry,
                project_points(polygon, evaluation.pred_homography),
                color,
            )
        _draw_keypoints(
            panel,
            geometry,
            prediction.keypoints.keypoints_xy,
            prediction.keypoints.valid,
            color,
        )
        median = evaluation.median_error_px
        covered = int((evaluation.gt_visible & evaluation.pred_valid).sum())
        summary = f"median={median:.2f}px" if median is not None else "median=n/a"
        iou = (
            "n/a" if evaluation.polygon_iou is None else f"{evaluation.polygon_iou:.3f}"
        )
        panels.append(
            _label(
                panel,
                (
                    f"{model}  {summary}  cov={covered}/{evaluation.gt_visible_count}"
                    f"  H={'ok' if evaluation.pred_homography is not None else 'FAIL'}"
                    f"  IoU={iou}"
                ),
            )
        )

    combined = _base_panel(sample)
    geometry = _geometry(sample, combined)
    drawn: list[str] = []
    if reference is not None and reference.gt_homography is not None:
        _draw_court_lines(
            combined, geometry, template_lines, reference.gt_homography, GT_COLOR
        )
        drawn.append("GT")
    for model, color in (("ours", OURS_COLOR), ("tcd", TCD_COLOR)):
        evaluation = evaluations.get(model)
        if evaluation is None or evaluation.pred_homography is None:
            continue
        _draw_court_lines(
            combined,
            geometry,
            template_lines,
            evaluation.pred_homography,
            color,
            thickness=1,
        )
        drawn.append(model)
    # Per-model IoU lives on each model panel, so the overlay cannot imply a
    # single score for several overlaid predictions.
    panels.append(_label(combined, f"court lines: {' + '.join(drawn)}"))
    return np.concatenate(panels, axis=1)


def render_domain_montage(
    rows: Sequence[tuple[SelectedSample, NDArray[np.uint8]]],
    *,
    title: str,
) -> NDArray[np.uint8]:
    """Stack labelled review rows into one montage image."""
    if not rows:
        raise ValueError("A montage requires at least one review row.")
    width = rows[0][1].shape[1]
    blocks: list[NDArray[np.uint8]] = [_header_banner(title, width=width)]
    for selected, image in rows:
        if image.shape[1] != width:
            raise ValueError("Every review row in a montage must share its width.")
        median = (
            "n/a"
            if selected.median_error_px is None
            else f"{selected.median_error_px:.2f}px"
        )
        blocks.append(
            _header_banner(
                (
                    f"{selected.sample_id} | {selected.domain} | scene={selected.scene_id}"
                    f" | {selected.primary_model} median={median}"
                    f" | alignment={'FAIL' if selected.alignment_failed else 'ok'}"
                    f" | selected by {', '.join(selected.selection_reasons)}"
                ),
                width=width,
                height=HEADER_HEIGHT,
            )
        )
        blocks.append(image)
    separator = np.full((TILE_GAP, width, 3), BACKGROUND, dtype=np.uint8)
    merged: list[NDArray[np.uint8]] = []
    for block in blocks:
        if merged:
            merged.append(separator)
        merged.append(block)
    return np.concatenate(merged, axis=0)


def render_error_cdf(
    path: Path,
    *,
    per_model_errors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    display_names: Mapping[str, str],
) -> None:
    """Plot the CDF of per-keypoint error for every domain and model."""
    domains = sorted(
        {domain for models in per_model_errors.values() for domain in models}
    )
    figure, axes = plt.subplots(
        1,
        max(1, len(domains)),
        figsize=(6.0 * max(1, len(domains)), 4.2),
        squeeze=False,
    )
    for column, domain in enumerate(domains):
        axis = axes[0][column]
        plotted = 0
        for model in MODEL_ORDER:
            values = per_model_errors.get(model, {}).get(domain)
            if values is None or values.size == 0:
                continue
            ordered = np.sort(values)
            cdf = np.arange(1, ordered.size + 1) / ordered.size
            axis.plot(ordered, cdf, label=f"{model} (n={ordered.size})")
            plotted += 1
        axis.set_title(display_names.get(domain, domain))
        axis.set_xlabel("keypoint error / image diagonal")
        axis.set_ylabel("cumulative fraction of valid pairs")
        axis.set_xlim(left=0.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
        if plotted:
            axis.legend(loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def render_summary_bars(
    path: Path,
    *,
    aggregates: Sequence[Mapping[str, object]],
    display_names: Mapping[str, str],
) -> None:
    """Plot the headline metrics per domain and model side by side."""
    panels = (
        ("completeness", "completeness", (0.0, 1.0)),
        ("PCK@0.02", "pck_002", (0.0, 1.0)),
        ("H success", "homography_success", (0.0, 1.0)),
        ("doubles IoU", "polygon_iou", (0.0, 1.0)),
    )
    domains = sorted({str(item["domain"]) for item in aggregates})
    if not domains:
        raise ValueError("Summary bars require at least one aggregate.")
    # Only plot models that actually produced an aggregate for this run: a model
    # that was not evaluated must not appear as a zero-height bar.
    models = sorted(
        {str(item["model"]) for item in aggregates},
        key=lambda model: (
            MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
        ),
    )
    figure, axes = plt.subplots(
        1, len(panels), figsize=(4.0 * len(panels), 4.0), squeeze=False
    )
    width = 0.36
    positions: NDArray[np.float64] = np.arange(len(domains), dtype=np.float64)
    for column, (title, key, limits) in enumerate(panels):
        axis = axes[0][column]
        for offset, model in enumerate(models):
            values = [
                _headline_value(aggregates, domain=domain, model=model, key=key)
                for domain in domains
            ]
            axis.bar(
                positions + (offset - (len(models) - 1) / 2.0) * width,
                [0.0 if value is None else value for value in values],
                width=width,
                label=model,
            )
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [display_names.get(domain, domain) for domain in domains], rotation=10
        )
        axis.set_ylim(*limits)
        axis.grid(alpha=0.3, axis="y")
        if column == 0:
            axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _headline_value(
    aggregates: Sequence[Mapping[str, object]],
    *,
    domain: str,
    model: str,
    key: str,
) -> float | None:
    for item in aggregates:
        if item["domain"] != domain or item["model"] != model:
            continue
        keypoints = item["keypoints"]
        alignment = item["alignment"]
        if not isinstance(keypoints, Mapping) or not isinstance(alignment, Mapping):
            raise ValueError("Aggregate groups must be mappings.")
        if key == "completeness":
            return _optional_float(keypoints["completeness"])
        if key == "pck_002":
            pck = keypoints["pck"]
            if not isinstance(pck, Mapping):
                raise ValueError("PCK aggregates must be mappings.")
            return _optional_float(_pck_by_value(pck, 0.02))
        if key == "homography_success":
            return _optional_float(alignment["predicted_homography_success_rate"])
        if key == "polygon_iou":
            iou = alignment["doubles_polygon_iou"]
            if not isinstance(iou, Mapping):
                raise ValueError("Polygon IoU aggregates must be mappings.")
            return _optional_float(iou["mean"])
    return None


def _pck_by_value(payload: Mapping[str, object], needle: float) -> object:
    for key, value in payload.items():
        try:
            if abs(float(key) - needle) < 1e-12:
                return value
        except (TypeError, ValueError):
            continue
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected a numeric metric value, got {type(value).__name__}.")


def _geometry(sample: LoadedSample, panel: NDArray[np.uint8]) -> PanelGeometry:
    return PanelGeometry(
        source_height=sample.ref.height,
        source_width=sample.ref.width,
        panel_height=panel.shape[0],
        panel_width=panel.shape[1],
    )


def _base_panel(sample: LoadedSample) -> NDArray[np.uint8]:
    image = np.asarray(sample.image_rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = PANEL_WIDTH / float(width)
    resized = cv2.resize(
        image,
        (PANEL_WIDTH, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized[:, :, ::-1])


def _draw_court_lines(
    panel: NDArray[np.uint8],
    geometry: PanelGeometry,
    template_lines: Sequence[NDArray[np.float64]],
    homography: NDArray[np.float64],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    """Draw each regulation line as its own polyline, never bridged together."""
    for template_line in template_lines:
        _polyline(
            panel,
            geometry.scale(project_points(template_line, homography)),
            color,
            thickness,
        )


def _draw_polygon(
    panel: NDArray[np.uint8],
    geometry: PanelGeometry,
    polygon: NDArray[np.float64],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    scaled = geometry.scale(polygon)
    _polyline(panel, np.concatenate([scaled, scaled[:1]], axis=0), color, thickness)


def _polyline(
    panel: NDArray[np.uint8],
    points: NDArray[np.float64],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for start, end in zip(points[:-1], points[1:], strict=True):
        if not (np.isfinite(start).all() and np.isfinite(end).all()):
            continue
        cv2.line(
            panel,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_keypoints(
    panel: NDArray[np.uint8],
    geometry: PanelGeometry,
    points: NDArray[np.float64],
    visible: NDArray[np.bool_],
    color: tuple[int, int, int],
) -> None:
    scaled = geometry.scale(points)
    for index in range(KEYPOINT_COUNT):
        if not visible[index] or not np.isfinite(scaled[index]).all():
            continue
        center = (int(round(scaled[index][0])), int(round(scaled[index][1])))
        cv2.circle(panel, center, KEYPOINT_RADIUS, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, center, KEYPOINT_RADIUS, BACKGROUND, 1, lineType=cv2.LINE_AA)


def _label(panel: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    banner = np.full((LABEL_HEIGHT, panel.shape[1], 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(
        banner,
        text,
        (6, LABEL_HEIGHT - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        lineType=cv2.LINE_AA,
    )
    return np.concatenate((banner, panel), axis=0)


def _header_banner(
    text: str, *, width: int, height: int = HEADER_HEIGHT
) -> NDArray[np.uint8]:
    banner: NDArray[np.uint8] = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(
        banner,
        text,
        (8, height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        lineType=cv2.LINE_AA,
    )
    return banner


__all__ = [
    "GT_COLOR",
    "MODEL_ORDER",
    "OURS_COLOR",
    "QUANTILE_LADDER",
    "TCD_COLOR",
    "PanelGeometry",
    "SelectedSample",
    "render_domain_montage",
    "render_error_cdf",
    "render_sample_row",
    "render_summary_bars",
    "select_review_samples",
]
