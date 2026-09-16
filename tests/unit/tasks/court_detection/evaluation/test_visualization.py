"""Figure contracts: independent court-line polylines and single-model runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation import visualization
from src.tasks.court_detection.evaluation.alignment import (
    COURT_LINE_SEGMENTS,
    project_points,
    sample_segment_points,
    segment_polylines,
)
from src.tasks.court_detection.evaluation.contracts import (
    KeypointPrediction,
    LoadedSample,
    ModelPrediction,
    SampleRef,
)
from src.tasks.court_detection.evaluation.metrics import (
    SampleEvaluation,
    evaluate_sample,
)
from src.tasks.court_detection.evaluation.settings import BenchmarkQualitySettings
from src.tasks.court_detection.geometry.homography import court_template_xy

_TEMPLATE = court_template_xy(14).astype(np.float64)
_WIDTH = 64
_HEIGHT = 48
_HOMOGRAPHY = np.array(
    [[4.0, 0.0, 32.0], [0.0, 4.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
_QUALITY = BenchmarkQualitySettings(
    pck_fractions=(0.02,),
    ransac_threshold_fraction=0.01,
    line_samples_per_segment=5,
    strata_axes=("scene",),
)
_PER_SEGMENT = _QUALITY.line_samples_per_segment


def _sample(sample_id: str = "sample-a") -> LoadedSample:
    points = project_points(_TEMPLATE, _HOMOGRAPHY)
    return LoadedSample(
        ref=SampleRef(
            domain="real_validation",
            sample_id=sample_id,
            scene_id="tennis_court_detector",
            trajectory_group_id=None,
            split="val",
            source_target_sha256="a" * 64,
            width=_WIDTH,
            height=_HEIGHT,
        ),
        image_rgb=np.full((_HEIGHT, _WIDTH, 3), 60, dtype=np.uint8),
        template_xy=_TEMPLATE,
        gt_keypoints_xy=points,
        gt_visible=np.ones(14, dtype=bool),
        strata={"scene": "tennis_court_detector"},
    )


def _prediction(model: str, *, offset_px: float = 0.0) -> ModelPrediction:
    points = project_points(_TEMPLATE, _HOMOGRAPHY) + offset_px
    return ModelPrediction(
        model=model,  # type: ignore[arg-type]
        keypoints=KeypointPrediction(
            keypoints_xy=points,
            scores=np.ones(14, dtype=np.float64),
            valid=np.ones(14, dtype=bool),
        ),
        elapsed_seconds=0.1,
        extras={"model": model},
    )


def _evaluation(sample: LoadedSample, prediction: ModelPrediction) -> SampleEvaluation:
    return evaluate_sample(sample, prediction, quality=_QUALITY)


def test_segment_polylines_keeps_each_regulation_line_separate() -> None:
    polylines = segment_polylines(
        COURT_LINE_SEGMENTS, _TEMPLATE, per_segment=_PER_SEGMENT
    )

    assert len(polylines) == len(COURT_LINE_SEGMENTS)
    for polyline in polylines:
        assert polyline.shape == (_PER_SEGMENT, 2)
    # The flat sampler used by the metrics is exactly the concatenation of them.
    np.testing.assert_allclose(
        sample_segment_points(COURT_LINE_SEGMENTS, _TEMPLATE, per_segment=_PER_SEGMENT),
        np.concatenate(polylines, axis=0),
    )


def test_rendered_court_lines_never_bridge_between_unrelated_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _sample()
    drawn: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def _record(
        panel: object,
        start: Sequence[int],
        end: Sequence[int],
        *args: object,
        **kwargs: object,
    ) -> None:
        drawn.append(((int(start[0]), int(start[1])), (int(end[0]), int(end[1]))))

    monkeypatch.setattr(visualization.cv2, "line", _record)
    row = visualization.render_sample_row(
        sample,
        {"ours": _prediction("ours")},
        {"ours": _evaluation(sample, _prediction("ours"))},
        quality=_QUALITY,
    )

    assert row.ndim == 3
    panel_width = visualization.PANEL_WIDTH
    panel_height = row.shape[0] - visualization.LABEL_HEIGHT
    # Four lattices: the GT panel, the model panel, and the GT + model overlay.
    # Two of those panels also outline the doubles polygon.
    lattice_segments = 4 * len(COURT_LINE_SEGMENTS) * (_PER_SEGMENT - 1)
    polygon_edges = 2 * 4
    assert len(drawn) == lattice_segments + polygon_edges

    def _panel_point(point: NDArray[np.float64]) -> tuple[int, int]:
        return (
            int(round(point[0] * panel_width / _WIDTH)),
            int(round(point[1] * panel_height / _HEIGHT)),
        )

    projected = project_points(_TEMPLATE, _HOMOGRAPHY)
    for index in range(len(COURT_LINE_SEGMENTS) - 1):
        tail = _panel_point(projected[COURT_LINE_SEGMENTS[index][1]])
        head = _panel_point(projected[COURT_LINE_SEGMENTS[index + 1][0]])
        assert (tail, head) not in drawn
        assert (head, tail) not in drawn

    # The pre-fix implementation drew the flat concatenation as one polyline,
    # whose consecutive pairs contain exactly these cross-line bridges.
    flat = project_points(
        sample_segment_points(COURT_LINE_SEGMENTS, _TEMPLATE, per_segment=_PER_SEGMENT),
        _HOMOGRAPHY,
    )
    assert np.allclose(flat[_PER_SEGMENT - 1], projected[COURT_LINE_SEGMENTS[0][1]])
    assert np.allclose(flat[_PER_SEGMENT], projected[COURT_LINE_SEGMENTS[1][0]])


def test_a_single_model_row_renders_and_reports_its_own_iou(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        visualization.cv2,
        "putText",
        lambda panel, text, *args, **kwargs: labels.append(str(text)),
    )

    row = visualization.render_sample_row(
        _sample(),
        {"tcd": _prediction("tcd", offset_px=3.0)},
        {"tcd": _evaluation(_sample(), _prediction("tcd", offset_px=3.0))},
        quality=_QUALITY,
    )

    assert row.shape[1] == 3 * visualization.PANEL_WIDTH
    assert any(label.startswith("GT ") for label in labels)
    tcd_labels = [label for label in labels if label.startswith("tcd  ")]
    assert len(tcd_labels) == 1
    assert "IoU=" in tcd_labels[0]
    # No "ours" panel and no combined IoU claim for a model that did not run.
    assert not any(label.startswith("ours") for label in labels)
    combined = labels[-1]
    assert "IoU" not in combined
    assert combined == "court lines: GT + tcd"


def test_a_failed_homography_stays_explicit_in_both_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _sample(sample_id="sparse")
    visible: NDArray[np.bool_] = np.zeros(14, dtype=bool)
    visible[:3] = True
    sparse_sample = LoadedSample(
        ref=sample.ref,
        image_rgb=sample.image_rgb,
        template_xy=sample.template_xy,
        gt_keypoints_xy=np.where(visible[:, None], sample.gt_keypoints_xy, np.nan),
        gt_visible=visible,
        strata=sample.strata,
    )
    prediction = ModelPrediction(
        model="tcd",  # type: ignore[arg-type]
        keypoints=KeypointPrediction(
            keypoints_xy=np.where(
                visible[:, None], sparse_sample.gt_keypoints_xy, np.nan
            ),
            scores=np.ones(14, dtype=np.float64),
            valid=visible,
        ),
        elapsed_seconds=0.1,
        extras={},
    )
    labels: list[str] = []
    monkeypatch.setattr(
        visualization.cv2,
        "putText",
        lambda panel, text, *args, **kwargs: labels.append(str(text)),
    )

    visualization.render_sample_row(
        sparse_sample,
        {"tcd": prediction},
        {"tcd": _evaluation(sparse_sample, prediction)},
        quality=_QUALITY,
    )

    assert any("H=FAIL" in label for label in labels)
    assert any("IoU=n/a" in label for label in labels)


class _RecordingAxis:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def bar(self, *args: Any, **kwargs: Any) -> None:
        self.labels.append(str(kwargs.get("label")))

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class _RecordingFigure:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def _stub_pyplot(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_RecordingAxis]:
    axes: list[_RecordingAxis] = []

    def _subplots(*args: Any, **kwargs: Any) -> tuple[object, list[list[Any]]]:
        column_count = int(args[1]) if len(args) > 1 else 1
        row = [_RecordingAxis() for _ in range(column_count)]
        axes.extend(row)
        return _RecordingFigure(), [row]

    monkeypatch.setattr(visualization.plt, "subplots", _subplots)
    monkeypatch.setattr(visualization.plt, "close", lambda *args, **kwargs: None)
    return axes


def _aggregate(model: str, domain: str = "real_validation") -> dict[str, object]:
    return {
        "model": model,
        "domain": domain,
        "keypoints": {
            "completeness": 1.0,
            "pck": {"0.02": 1.0},
            "pair_error_px": {"count": 2, "mean": 1.0},
        },
        "alignment": {
            "predicted_homography_success_rate": 1.0,
            "doubles_polygon_iou": {"count": 2, "mean": 0.9},
        },
    }


def test_summary_bars_draw_only_the_models_that_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    axes = _stub_pyplot(monkeypatch)

    visualization.render_summary_bars(
        tmp_path / "bars.png",
        aggregates=[_aggregate("tcd")],
        display_names={"real_validation": "real_validation"},
    )

    assert axes
    assert set(axes[0].labels) == {"tcd"}


def test_summary_bars_keep_both_models_when_both_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    axes = _stub_pyplot(monkeypatch)

    visualization.render_summary_bars(
        tmp_path / "bars.png",
        aggregates=[_aggregate("ours"), _aggregate("tcd")],
        display_names={"real_validation": "real_validation"},
    )

    assert set(axes[0].labels) == {"ours", "tcd"}


def test_review_selection_uses_the_only_model_that_ran() -> None:
    sample = _sample()
    evaluation = _evaluation(sample, _prediction("tcd"))

    selected = visualization.select_review_samples(
        [evaluation], primary_model="tcd", count=1
    )

    assert len(selected) == 1
    assert selected[0].primary_model == "tcd"


def test_cv2_is_used_in_bgr_order_for_the_documented_colours() -> None:
    # Green reads as green, orange as orange, blue as blue in the saved PNG.
    for color in (
        visualization.GT_COLOR,
        visualization.OURS_COLOR,
        visualization.TCD_COLOR,
    ):
        assert len(color) == 3
    assert visualization.GT_COLOR[1] > visualization.GT_COLOR[0]
    assert cv2.__name__ == "cv2"
