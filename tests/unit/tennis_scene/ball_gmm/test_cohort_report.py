"""The comparison-only cohort script must join camera AND frame, not row order."""

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path("knowledge/runs/run-ball3d-candidate-gmm-20261103/cohorts.py")


@pytest.fixture(scope="module")
def cohort_script() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT))


def test_camera_frame_partition_errors_and_split_boundary_match_hand_values(
    cohort_script: dict[str, Any],
) -> None:
    baseline = {
        ("cam2", 504): (3.0, "observed"),
        ("cam0", 505): (1.0, "observed"),
        ("cam1", 505): (21.0, "estimated"),
        ("cam1", 506): (5.0, "observed"),
        ("cam2", 505): (9.0, "observed"),
    }
    method = {
        ("cam2", 506): (41.0, "observed"),
        ("cam1", 505): (31.0, "estimated"),
        ("cam0", 506): (20.0, "observed"),
        ("cam2", 504): (13.0, "observed"),
        ("cam0", 505): (11.0, "observed"),
    }
    summarize = cohort_script["cohort_metrics"]
    result = summarize(method, baseline, start=505, stop=1010)
    for key, median, p90, rate in (
        ("common_method", 21.0, 29.0, 0.5),
        ("common_baseline", 11.0, 19.0, 0.5),
        ("method_only", 30.5, 38.9, 0.5),
        ("baseline_only", 7.0, 8.6, 0.0),
    ):
        assert result[key]["count"] == 2
        assert result[key]["median"] == pytest.approx(median)
        assert result[key]["p90"] == pytest.approx(p90)
        assert result[key]["above_threshold_rate"] == rate
    assert result["method_all"]["count"] == result["baseline_all"]["count"] == 4
    tuning = summarize(method, baseline, start=0, stop=505)
    assert tuning["common_method"]["count"] == 1
    assert tuning["common_method"]["median"] == 13
    estimated = summarize(
        method, baseline, start=505, stop=1010, camera="cam1", kind="estimated"
    )
    assert estimated["common_method"]["count"] == 1
    assert estimated["common_method"]["median"] == 31
    empty = summarize(
        method, baseline, start=505, stop=1010, camera="cam2", kind="estimated"
    )
    assert empty["method_all"]["count"] == 0
    assert empty["method_all"]["median"] is None
    assert empty["method_all"]["above_threshold_rate"] is None


def test_duplicate_lovo_identity_is_rejected(cohort_script: dict[str, Any]) -> None:
    samples = {
        "camera_index": np.array([1, 1]),
        "frame_index": np.array([505, 505]),
        "error_px": np.array([7.0, 17.0]),
        "reference_kind": np.array(["observed", "observed"]),
    }
    with pytest.raises(ValueError, match="Duplicate LOVO camera/frame"):
        cohort_script["index_samples"](samples, ("cam0", "cam1", "cam2"), "cam1")


def test_common_identity_cannot_change_reference_kind(
    cohort_script: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="inconsistent reference kind"):
        cohort_script["partitions"](
            {("cam1", 505): (7.0, "observed")}, {("cam1", 505): (17.0, "estimated")}
        )
