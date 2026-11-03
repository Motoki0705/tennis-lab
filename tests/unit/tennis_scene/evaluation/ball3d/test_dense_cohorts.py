"""Camera/frame cohort membership and errors, independently hand-calculated."""

import importlib.util
import sys

import numpy as np
import pytest

from src.utils.paths import PROJECT_ROOT

_PATH = PROJECT_ROOT / "knowledge/runs/run-ball3d-dense-likelihood-20261103/cohorts.py"
_SPEC = importlib.util.spec_from_file_location("dense_likelihood_cohorts", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
cohorts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cohorts
_SPEC.loader.exec_module(cohorts)


def asymmetric_samples():
    point = cohorts.Sample
    baseline = {("cam2", 505): point(1, "observed"), ("cam0", 505): point(9, "observed"),
                ("cam0", 504): point(3, "estimated"), ("cam1", 1009): point(25, "estimated"),
                ("cam2", 504): point(20, "observed")}
    method = {("cam1", 504): point(2, "estimated"), ("cam2", 1009): point(100, "estimated"),
              ("cam1", 506): point(20, "observed"), ("cam0", 504): point(7, "estimated"),
              ("cam0", 505): point(5, "observed"), ("cam2", 505): point(41, "observed")}
    return baseline, method


def test_join_preserves_camera_identity_and_split_boundary_and_error_owners():
    baseline, method = asymmetric_samples()
    row = cohorts.partition(baseline, method, 505, 1010)
    assert row["counts"] == {"common": 2, "method_only": 2, "baseline_only": 1, "method_all": 4, "baseline_all": 3}
    for value, median, p90, rate in ((row["common"]["method"], 23, 37.4, .5),
                                    (row["common"]["baseline"], 5, 8.2, 0),
                                    (row["method_only"]["method"], 60, 92, .5),
                                    (row["baseline_only"]["baseline"], 25, 25, 1)):
        assert value["median"] == pytest.approx(median)
        assert value["p90"] == pytest.approx(p90)
        assert value["above_threshold_rate"] == rate
    assert row["method_only"]["method"]["above_threshold_count"] == 1  # 20px itself is excluded.
    tuning = cohorts.partition(baseline, method, 0, 505)
    assert tuning["counts"] == {"common": 1, "method_only": 1, "baseline_only": 1, "method_all": 2, "baseline_all": 2}


def test_camera_kind_breakdowns_and_empty_cohorts_are_unmeasured():
    baseline, method = asymmetric_samples()
    rows = cohorts.all_rows(baseline, method, ("cam0", "cam1", "cam2"))
    chosen = next(r for r in rows if r["segment"] == "evaluation" and r["held_out_camera"] == "cam2" and r["reference_kind"] == "estimated")
    assert chosen["counts"] == {"common": 0, "method_only": 1, "baseline_only": 0, "method_all": 1, "baseline_all": 0}
    assert chosen["method_only"]["method"]["median"] == 100
    for owner in ("method", "baseline"):
        assert chosen["common"][owner]["count"] == 0
        assert chosen["common"][owner]["median"] is None
        assert chosen["common"][owner]["p90"] is None
        assert chosen["common"][owner]["above_threshold_rate"] is None
    identity = cohorts.partition(baseline, baseline, 0, 1010)
    assert identity["counts"]["common"] == 5
    assert identity["method_only"]["method"]["count"] == 0
    assert identity["baseline_only"]["baseline"]["above_threshold_rate"] is None


@pytest.mark.parametrize("fault", ["duplicate", "wrong_camera", "kind_mismatch"])
def test_ambiguous_sample_identity_stops_instead_of_silently_joining(fault):
    if fault == "kind_mismatch":
        baseline, method = asymmetric_samples()
        method["cam2", 505] = cohorts.Sample(41, "estimated")
        with pytest.raises(ValueError, match="Reference kind mismatch"):
            cohorts.partition(baseline, method, 505, 1010)
        return
    samples = {"camera_index": np.array([1]), "frame_index": np.array([505]),
               "error_px": np.array([3.]), "reference_kind": np.array(["observed"])}
    target: dict[tuple[str, int], object] = {}
    if fault == "duplicate":
        cohorts.merge_samples(target, samples, "cam1", ("cam0", "cam1", "cam2"))
    else:
        samples["camera_index"] = np.array([0])
    with pytest.raises(ValueError, match="Duplicate camera/frame|differs from the held-out"):
        cohorts.merge_samples(target, samples, "cam1", ("cam0", "cam1", "cam2"))
