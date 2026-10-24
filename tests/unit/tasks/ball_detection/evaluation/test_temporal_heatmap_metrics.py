"""Protocol arithmetic, indivisible score ties and endpoint calibration."""

import pytest

from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import (
    calibrate,
    frame_metrics,
    present,
)


def test_validation_calibration_groups_tied_scores_and_uses_lower_fpr_tie_break() -> None:
    records = [dict(kind=kind, score=.5) for kind in ("observed", "observed", "reviewed_empty", "out_of_frame")]
    selected = calibrate(records, split="val")
    assert selected["mode"] == "all_absent" and selected["threshold"] is None
    assert selected["false_positive"] == 0 and selected["false_negative"] == 2
    assert selected["balanced_error"] == .5 and not present(.5, selected)
    with pytest.raises(ValueError, match="only allowed on validation"):
        calibrate(records, split="test")
    with pytest.raises(ValueError, match="both observed"):
        calibrate(records[:2], split="val")


def test_validation_threshold_is_an_actual_score_change_point() -> None:
    records = [dict(kind="observed", score=.8), dict(kind="observed", score=.9),
               dict(kind="reviewed_empty", score=.2), dict(kind="out_of_frame", score=.5), dict(kind="masked", score=.95)]
    selected = calibrate(records, split="val")
    assert selected["threshold"] == .8 and selected["mode"] == "threshold" and selected["balanced_error"] == 0.
    assert present(.8, selected) and not present(.79, selected)


def test_location_errors_are_not_gated_and_empty_support_is_null() -> None:
    records = [dict(kind="observed", score=0., error_px=error) for error in (0., 20., 40., 100.)]
    records += [dict(kind="reviewed_empty", score=.6, error_px=None), dict(kind="out_of_frame", score=.3, error_px=None)]
    calibration = dict(mode="threshold", threshold=.5)
    report = frame_metrics(records, calibration, distance_px=20.)
    assert report["mean_error_px"] == 40. and report["median_error_px"] == 30.
    assert report["p95_error_px"] == pytest.approx(91.) and report["within_distance_rate"] == .5
    assert report["observed_fnr"] == 1. and report["negative_fpr"] == .5
    assert report["by_negative_kind"]["reviewed_empty"]["fpr"] == 1.
    assert report["by_negative_kind"]["out_of_frame"]["fpr"] == 0.
    empty = frame_metrics([], calibration, distance_px=20.)
    assert empty["mean_error_px"] is None and empty["observed_fnr"] is None and empty["negative_fpr"] is None
