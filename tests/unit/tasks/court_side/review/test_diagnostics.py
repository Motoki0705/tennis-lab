from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.court_side.review.diagnostics import SavedDiagnostics
from src.tasks.court_side.review.store import StoreCase


def test_saved_scores_and_non_scored_states(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    diagnosis = SavedDiagnostics(saved_diagnostics, StoreCase(stored_case))
    assert diagnosis.frame(2)["scores"]["costs"] == [0.2, 0.8]
    assert diagnosis.frame(2)["scores"]["cameras"] == ["cam0", "cam1"]
    assert diagnosis.frame(0) == {"state": "fewer_than_two_views", "scores": None}
    assert diagnosis.frame(1) == {"state": "not_sampled", "scores": None}


def test_csv_cannot_reorder_or_invent_support_frames(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    csv = saved_diagnostics / "production-frames.csv"
    csv.write_text(csv.read_text().replace("\n2,3,", "\n1,3,"))
    with pytest.raises(ValueError, match="CSV frame axis"):
        SavedDiagnostics(saved_diagnostics, StoreCase(stored_case))


def test_csv_scores_must_match_saved_aggregate(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    csv = saved_diagnostics / "production-frames.csv"
    csv.write_text(csv.read_text().replace("0.2,0.8", "0.3,0.8"))
    with pytest.raises(ValueError, match="aggregate"):
        SavedDiagnostics(saved_diagnostics, StoreCase(stored_case))


def test_other_observations_are_rejected(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    path = saved_diagnostics / "production-observations.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["uv_px"] = arrays["uv_px"] + 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="observations differ"):
        SavedDiagnostics(saved_diagnostics, StoreCase(stored_case))


def test_changed_execute_receipt_is_rejected(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    with (saved_diagnostics / "execute.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="input checksum"):
        SavedDiagnostics(saved_diagnostics, StoreCase(stored_case))


def test_changed_calibration_lineage_is_rejected(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    case = StoreCase(stored_case)
    case.references["court_calibration"] = {
        **case.references["court_calibration"],
        "artifact_id": "other-calibration",
    }
    with pytest.raises(ValueError, match="artifact lineage"):
        SavedDiagnostics(saved_diagnostics, case)


def test_other_source_is_rejected(stored_case: Path, saved_diagnostics: Path) -> None:
    case = StoreCase(stored_case)
    case.source = {**case.source, "clip_id": "other-clip"}
    with pytest.raises(ValueError, match="source/store"):
        SavedDiagnostics(saved_diagnostics, case)


def test_changed_report_hypotheses_are_rejected(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    path = saved_diagnostics / "production.json"
    report = json.loads(path.read_text())
    report["hypotheses"][0]["cost"] = 0.3
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="execution disagree"):
        SavedDiagnostics(saved_diagnostics, StoreCase(stored_case))
