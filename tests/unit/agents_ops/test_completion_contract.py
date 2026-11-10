"""No missing, unverified or exported obligation may pass as completion."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("parent.completion")


def finish() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "completed",
        "basis_ref": "fixed-agreement",
        "acceptance": [
            {"id": "agreement", "state": "met", "evidence": "合意した実機確認を実施。"},
            {"id": "ac-1", "state": "met", "evidence": "条件Aを入力して結果Bを確認。"},
        ],
        "remaining_required": [],
        "improvements": {"reviewed": True, "candidates": []},
        "summary": "全体受入済み。",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "omitted",
        "duplicate",
        "unmet",
        "unverified",
        "evidence",
        "remaining",
        "wrong_basis",
        "not_reviewed",
        "unknown_candidate",
        "exported",
    ],
)
def test_completion_refuses_each_missing_obligation(api: Any, fault: str) -> None:
    spec = finish()
    candidates: dict[str, Any] = {}
    if fault == "omitted":
        spec["acceptance"].pop()
    elif fault == "duplicate":
        spec["acceptance"].append(spec["acceptance"][-1])
    elif fault in {"unmet", "unverified"}:
        spec["acceptance"][-1]["state"] = fault
    elif fault == "evidence":
        spec["acceptance"][-1]["evidence"] = " "
    elif fault == "remaining":
        spec["remaining_required"] = ["必須の実機確認が未実施。"]
    elif fault == "wrong_basis":
        spec["basis_ref"] = "another-agreement"
    elif fault == "not_reviewed":
        spec["improvements"]["reviewed"] = False
    elif fault == "unknown_candidate":
        spec["improvements"]["candidates"] = ["I-1-unknown"]
    else:
        spec["improvements"]["candidates"] = ["I-1-later"]
        candidates["later"] = {
            "candidate": {"data": {"spec": {"unmet_acceptance": ["ac-1"]}}}
        }
    with pytest.raises(ValueError):
        api.validate_finish(
            spec,
            {
                "basis_ref": "fixed-agreement",
                "criteria": [{"id": "agreement"}, {"id": "ac-1"}],
            },
            candidates,
            1,
        )


def test_no_candidates_is_normal_and_incomplete_is_not_completed(api: Any) -> None:
    spec = finish()
    required = {
        "basis_ref": "fixed-agreement",
        "criteria": [{"id": "agreement"}, {"id": "ac-1"}],
    }
    api.validate_finish(spec, required, {}, 1)
    spec["state"] = "incomplete"
    spec["acceptance"][-1]["state"] = "unmet"
    spec["remaining_required"] = ["必須作業を元親に残す。"]
    api.validate_finish(spec, required, {}, 1)


@pytest.mark.parametrize(
    "fault", ["missing", "job", "generation", "authorization", "response", "finish"]
)
def test_parent_final_requires_the_exact_report_receipt(api: Any, fault: str) -> None:
    job = {"id": "runtime-1", "authorization": {"generation": 1}}
    response = {
        "schema_version": 1,
        "state": "completed",
        "reason": "受入済み",
        "resume_condition": "なし",
    }
    receipt = {
        "job": "runtime-1",
        "generation": 1,
        "authorization": job["authorization"],
        "response_sha256": api.digest(response),
        "finish_sha256": "finish-1",
    }
    row = {
        "generation": 1,
        "completion": {"state": "completed", "finish_sha256": "finish-1"},
        "development_result": {"completion_receipt": receipt},
    }
    api.require_report_receipt(row, job, response)
    if fault == "missing":
        row.pop("development_result")
    else:
        field = {"response": "response_sha256", "finish": "finish_sha256"}.get(
            fault, fault
        )
        receipt[field] = "different"
    with pytest.raises(ValueError, match="matching workflow-report"):
        api.require_report_receipt(row, job, response)
