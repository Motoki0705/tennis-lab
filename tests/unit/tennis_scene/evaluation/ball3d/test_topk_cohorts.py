"""Camera/frame cohort accounting for the saved top-K comparison report."""

import runpy

import numpy as np
import pytest

from src.utils.paths import PROJECT_ROOT

_REPORT = runpy.run_path(str(PROJECT_ROOT / "knowledge/runs/run-ball3d-topk-multihypothesis-20261103/cohort_report.py"))


def samples(rows):
    camera, frame, error, kind = zip(*rows, strict=True)
    return {"camera_index": np.asarray(camera, dtype=np.int64), "frame_index": np.asarray(frame, dtype=np.int64),
            "error_px": np.asarray(error, dtype=np.float64), "reference_kind": np.asarray(kind)}


def test_camera_frame_join_and_strata_preserve_reordered_asymmetric_points():
    ids = ("sideline", "rear", "high")
    method = _REPORT["index_samples"](samples([(0, 504, 4, "observed"), (0, 505, 5, "observed"),
        (1, 505, 24, "estimated"), (0, 1009, 120, "observed"), (2, 506, 20, "estimated")]), ids, 1010)
    baseline = _REPORT["index_samples"](samples([(2, 505, 0.5, "observed"), (1, 505, 2, "estimated"),
        (0, 505, 15, "observed"), (0, 504, 14, "observed")]), ids, 1010)
    rows = _REPORT["summarize"](method, baseline, ids)
    indexed = {(r["segment"], r["held_out_camera"], r["reference_kind"]): r for r in rows}
    evaluation = indexed["evaluation", "all", "all"]
    assert evaluation["common"]["count"] == 2
    assert evaluation["method_only"]["count"] == 2
    assert evaluation["baseline_only"]["count"] == 1
    assert evaluation["common"]["method"]["median"] == pytest.approx(14.5)
    assert evaluation["common"]["method"]["p90"] == pytest.approx(22.1)
    assert evaluation["common"]["baseline"]["median"] == pytest.approx(8.5)
    assert evaluation["common"]["baseline"]["p90"] == pytest.approx(13.7)
    assert evaluation["method_only"]["method"]["median"] == pytest.approx(70)
    assert evaluation["method_only"]["method"]["p90"] == pytest.approx(110)
    assert evaluation["method_only"]["method"]["above_threshold_count"] == 1  # exactly 20 is retained, not >20
    assert evaluation["method_only"]["method"]["above_threshold_rate"] == pytest.approx(0.5)
    assert evaluation["baseline_only"]["baseline"]["median"] == pytest.approx(0.5)
    assert evaluation["totals"]["method"]["count"] == 4
    assert evaluation["totals"]["baseline"]["count"] == 3
    high = indexed["evaluation", "high", "all"]
    assert high["common"]["count"] == 0  # same frame 505 in other cameras cannot join this camera
    assert high["method_only"]["count"] == high["baseline_only"]["count"] == 1
    estimated = indexed["evaluation", "all", "estimated"]
    assert estimated["common"]["count"] == estimated["method_only"]["count"] == 1
    assert estimated["baseline_only"]["count"] == 0
    tuning = indexed["tuning", "all", "all"]
    assert tuning["common"]["count"] == 1
    assert tuning["method_only"]["method"]["count"] == 0
    assert tuning["method_only"]["method"]["median"] is None
    assert tuning["method_only"]["method"]["p90"] is None
    assert tuning["method_only"]["method"]["above_threshold_rate"] is None
    assert indexed["all", "all", "all"]["common"]["count"] == 3
    assert all(all(row["checks"].values()) for row in rows)
    for cohort in ("common", "method_only", "baseline_only"):
        assert sum(indexed["evaluation", c, "all"][cohort]["count"] for c in ids) == evaluation[cohort]["count"]
        assert sum(indexed["evaluation", "all", k][cohort]["count"] for k in ("observed", "estimated")) == evaluation[cohort]["count"]


def test_duplicate_camera_frame_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        _REPORT["index_samples"](samples([(0, 505, 2, "observed"), (0, 505, 18, "observed")]), ("sideline",), 1010)


def test_common_reference_kind_mismatch_is_rejected_before_stratification():
    key = "sideline", 505
    with pytest.raises(ValueError, match="reference kind mismatch"):
        _REPORT["summarize"]({key: (2, "observed")}, {key: (18, "estimated")}, ("sideline",))


@pytest.mark.parametrize("totals", [(4, 3), (3, 4)])
def test_incomplete_partition_counts_stop_the_report(totals):
    with pytest.raises(ValueError, match="conserve"):
        _REPORT["validate_counts"](2, 1, 1, *totals)
