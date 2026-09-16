"""Denominator and undefined-value contracts for the benchmark metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.alignment import (
    ALIGNMENT_REASON_INSUFFICIENT_POINTS,
    project_points,
)
from src.tasks.court_detection.evaluation.contracts import (
    KeypointPrediction,
    LoadedSample,
    ModelPrediction,
    SampleRef,
)
from src.tasks.court_detection.evaluation.metrics import (
    REASON_NO_PAIR,
    aggregate_evaluations,
    evaluate_sample,
)
from src.tasks.court_detection.evaluation.settings import BenchmarkQualitySettings
from src.tasks.court_detection.geometry.homography import court_template_xy


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Narrow one aggregate into an indexable mapping for assertions."""
    assert isinstance(value, Mapping)
    return cast("Mapping[str, Any]", value)


_TEMPLATE = court_template_xy(14).astype(np.float64)
_HOMOGRAPHY = np.array([[6.0, 0.0, 320.0], [0.0, 6.0, 220.0], [0.0, 0.0, 1.0]])
_QUALITY = BenchmarkQualitySettings(
    pck_fractions=(0.005, 0.01, 0.02, 0.05),
    ransac_threshold_fraction=0.01,
    line_samples_per_segment=11,
    strata_axes=("scene",),
)


def _sample(
    *,
    visible: np.ndarray | None = None,
    scene_id: str = "tennis_court_detector",
    sample_id: str = "sample-a",
    domain: str = "real_validation",
    homography: np.ndarray | None = None,
) -> LoadedSample:
    points = project_points(
        _TEMPLATE, _HOMOGRAPHY if homography is None else np.asarray(homography)
    )
    mask = (
        np.ones(14, dtype=bool) if visible is None else np.asarray(visible, dtype=bool)
    )
    return LoadedSample(
        ref=SampleRef(
            domain=domain,  # type: ignore[arg-type]
            sample_id=sample_id,
            scene_id=scene_id,
            trajectory_group_id=None,
            split="val" if domain == "real_validation" else "test",
            source_target_sha256="a" * 64,
            width=960,
            height=540,
        ),
        image_rgb=np.zeros((540, 960, 3), dtype=np.uint8),
        template_xy=_TEMPLATE,
        gt_keypoints_xy=points,
        gt_visible=mask,
        strata={"scene": scene_id},
    )


def _prediction(
    *,
    keypoints: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    model: str = "ours",
) -> ModelPrediction:
    points = (
        project_points(_TEMPLATE, _HOMOGRAPHY)
        if keypoints is None
        else np.asarray(keypoints, dtype=np.float64)
    )
    mask = np.ones(14, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return ModelPrediction(
        model=model,  # type: ignore[arg-type]
        keypoints=KeypointPrediction(
            keypoints_xy=np.where(mask[:, None], points, np.nan),
            scores=np.ones(14, dtype=np.float64),
            valid=mask,
        ),
        elapsed_seconds=0.25,
        extras={},
    )


def test_a_perfect_prediction_scores_zero_error_and_full_pck() -> None:
    evaluation = evaluate_sample(_sample(), _prediction(), quality=_QUALITY)
    aggregate = _as_mapping(aggregate_evaluations([evaluation], quality=_QUALITY))

    keypoints = aggregate["keypoints"]
    assert keypoints["ground_truth_visible_keypoints"] == 14
    assert keypoints["completeness"] == 1.0
    assert keypoints["pair_error_px"]["count"] == 14
    assert keypoints["pair_error_px"]["median"] == pytest.approx(0.0)
    assert all(value == 1.0 for value in keypoints["pck"].values())
    alignment = aggregate["alignment"]
    assert alignment["predicted_homography_success_rate"] == 1.0
    assert alignment["line_reprojection_px"]["mean"] == pytest.approx(0.0)
    assert alignment["doubles_polygon_iou"]["mean"] == pytest.approx(1.0)
    assert alignment["undefined_reasons"] == {}


def test_missing_predictions_count_as_pck_misses_and_reduce_completeness() -> None:
    valid: NDArray[np.bool_] = np.ones(14, dtype=bool)
    valid[4:] = False
    evaluation = evaluate_sample(_sample(), _prediction(valid=valid), quality=_QUALITY)
    aggregate = _as_mapping(aggregate_evaluations([evaluation], quality=_QUALITY))

    keypoints = aggregate["keypoints"]
    assert keypoints["ground_truth_visible_keypoints"] == 14
    assert keypoints["predicted_valid_keypoints_among_visible"] == 4
    assert keypoints["completeness"] == pytest.approx(4 / 14)
    assert keypoints["pck"]["0.02"] == pytest.approx(4 / 14)
    # Only the four paired keypoints contribute to the error distribution.
    assert keypoints["pair_error_px"]["count"] == 4


def test_an_offset_prediction_reports_the_offset_in_pixels_and_diagonal() -> None:
    shifted = project_points(_TEMPLATE, _HOMOGRAPHY) + np.array([12.0, 0.0])
    evaluation = evaluate_sample(
        _sample(), _prediction(keypoints=shifted), quality=_QUALITY
    )
    aggregate = _as_mapping(aggregate_evaluations([evaluation], quality=_QUALITY))

    keypoints = aggregate["keypoints"]
    assert keypoints["pair_error_px"]["median"] == pytest.approx(12.0)
    diagonal = float(np.hypot(960, 540))
    assert keypoints["pair_error_diagonal"]["median"] == pytest.approx(12.0 / diagonal)
    assert keypoints["pck"]["0.005"] == pytest.approx(0.0)
    assert keypoints["pck"]["0.05"] == pytest.approx(1.0)


def test_homography_failure_is_reported_with_its_reason_and_zero_success() -> None:
    valid: NDArray[np.bool_] = np.zeros(14, dtype=bool)
    valid[:3] = True
    evaluation = evaluate_sample(
        _sample(visible=valid), _prediction(valid=valid), quality=_QUALITY
    )
    aggregate = _as_mapping(aggregate_evaluations([evaluation], quality=_QUALITY))

    alignment = aggregate["alignment"]
    assert alignment["predicted_homography_success"] == 0
    assert alignment["predicted_homography_success_rate"] == 0.0
    assert alignment["ground_truth_homography_success"] == 0
    assert alignment["predicted_homography_failure_reasons"] == {
        ALIGNMENT_REASON_INSUFFICIENT_POINTS: 1
    }
    # A failed fit must not contribute a zero inlier count to the statistics.
    assert alignment["predicted_homography_inliers_on_success"]["count"] == 0
    assert alignment["predicted_homography_inliers_on_success"]["mean"] is None
    assert alignment["undefined_reasons"][REASON_NO_PAIR] == 1
    assert alignment["line_reprojection_px"]["count"] == 0
    assert alignment["line_reprojection_px"]["mean"] is None
    assert alignment["doubles_polygon_iou"]["mean"] is None


def test_aggregate_reports_per_channel_and_strata_breakdowns() -> None:
    first = evaluate_sample(_sample(sample_id="s1"), _prediction(), quality=_QUALITY)
    second = evaluate_sample(
        _sample(sample_id="s2", scene_id="B00"),
        _prediction(model="tcd"),
        quality=_QUALITY,
    )
    aggregate = _as_mapping(aggregate_evaluations([first], quality=_QUALITY))
    second_aggregate = _as_mapping(aggregate_evaluations([second], quality=_QUALITY))

    assert aggregate["strata"]["scene"]["tennis_court_detector"]["sample_count"] == 1
    assert second_aggregate["strata"]["scene"]["B00"]["sample_count"] == 1
    channels = aggregate["per_channel"]
    assert len(channels) == 14
    assert channels[0]["name"] == "far_doubles_left"
    assert channels[0]["pck"]["0.01"] == 1.0
    assert channels[0]["pair_error_px"]["count"] == 1


def test_aggregation_refuses_to_mix_models_or_domains() -> None:
    ours = evaluate_sample(_sample(), _prediction(), quality=_QUALITY)
    tcd = evaluate_sample(_sample(), _prediction(model="tcd"), quality=_QUALITY)

    with pytest.raises(ValueError, match="exactly one model and one domain"):
        aggregate_evaluations([ours, tcd], quality=_QUALITY)


def test_metrics_are_undefined_rather_than_zero_without_visible_ground_truth() -> None:
    invisible: NDArray[np.bool_] = np.zeros(14, dtype=bool)
    evaluation = evaluate_sample(
        _sample(visible=invisible),
        _prediction(valid=invisible),
        quality=_QUALITY,
    )
    aggregate = _as_mapping(aggregate_evaluations([evaluation], quality=_QUALITY))

    keypoints = aggregate["keypoints"]
    assert keypoints["ground_truth_visible_keypoints"] == 0
    assert keypoints["completeness"] is None
    assert all(value is None for value in keypoints["pck"].values())
    assert keypoints["pair_error_px"]["count"] == 0


def test_line_error_is_published_for_the_whole_court_and_for_the_visible_frame() -> (
    None
):
    # A court that only partially fits the frame: the in-frame subset is a strict
    # subset of the sampled lines, and both metrics stay at zero for an exact
    # prediction.
    zoomed = np.array([[30.0, 0.0, 320.0], [0.0, 30.0, 100.0], [0.0, 0.0, 1.0]])
    points = project_points(_TEMPLATE, zoomed)
    evaluation = evaluate_sample(
        _sample(homography=zoomed), _prediction(keypoints=points), quality=_QUALITY
    )
    aggregate = _as_mapping(aggregate_evaluations([evaluation], quality=_QUALITY))

    alignment = _as_mapping(aggregate["alignment"])
    assert alignment["line_reprojection_px"]["mean"] == pytest.approx(0.0)
    assert alignment["line_reprojection_in_frame_px"]["mean"] == pytest.approx(0.0)
    assert 0 < alignment["line_samples_in_frame"] < alignment["line_samples_total"]
    assert 0.0 < alignment["line_samples_in_frame_fraction"] < 1.0


def test_in_frame_line_error_isolates_extrapolation_from_in_frame_error() -> None:
    # A shifted prediction misaligns every sampled line equally, so both the
    # whole-court and in-frame metrics report the same offset.
    shifted = _HOMOGRAPHY.copy()
    shifted[0, 2] += 9.0
    evaluation = evaluate_sample(
        _sample(),
        _prediction(keypoints=project_points(_TEMPLATE, shifted)),
        quality=_QUALITY,
    )
    aggregate = _as_mapping(aggregate_evaluations([evaluation], quality=_QUALITY))

    alignment = _as_mapping(aggregate["alignment"])
    assert alignment["line_reprojection_px"]["mean"] == pytest.approx(9.0)
    assert alignment["line_reprojection_in_frame_px"]["mean"] == pytest.approx(9.0)


def test_successful_homography_inliers_average_only_the_successful_fits() -> None:
    # Two samples: one perfect fit (14 inliers) and one with too few points to
    # fit at all.  The failure is reported through the success rate, not as a
    # zero inlier count that would halve the mean.
    success = evaluate_sample(_sample(sample_id="ok"), _prediction(), quality=_QUALITY)
    sparse: NDArray[np.bool_] = np.zeros(14, dtype=bool)
    sparse[:3] = True
    failure = evaluate_sample(
        _sample(sample_id="sparse", visible=sparse),
        _prediction(valid=sparse),
        quality=_QUALITY,
    )
    aggregate = _as_mapping(aggregate_evaluations([success, failure], quality=_QUALITY))

    alignment = _as_mapping(aggregate["alignment"])
    stats = _as_mapping(alignment["predicted_homography_inliers_on_success"])
    assert alignment["predicted_homography_success"] == 1
    assert alignment["sample_count"] == 2
    assert alignment["predicted_homography_success_rate"] == pytest.approx(0.5)
    assert stats["count"] == 1
    assert stats["mean"] == pytest.approx(14.0)
    assert stats["median"] == pytest.approx(14.0)
