"""A successful Actions check must match the pinned PR and test merge."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
HEAD = "a" * 40
BASE = "b" * 40
MERGE = "c" * 40


@pytest.fixture
def workflow_run() -> dict[str, object]:
    return {
        "path": ".github/workflows/knowledge.yml",
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "head_sha": HEAD,
        "id": 141,
        "run_attempt": 2,
    }


@pytest.fixture
def record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "records-only",
        "pr_number": 12,
        "head_sha": HEAD,
        "base_sha": BASE,
        "merge_sha": MERGE,
        "run_id": 141,
        "run_attempt": 2,
        "checks": {"kg_validate": "success", "kg_repro_paths": "success"},
        "conclusion": "success",
    }


@pytest.fixture
def verify(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("lib.knowledge_ci").verify_result


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("head_sha", "d" * 40, "head_sha does not match"),
        ("base_sha", "d" * 40, "base_sha does not match"),
        pytest.param(
            "merge_sha",
            "d" * 40,
            "merge_sha does not match",
            id="different-test-merge-sha",
        ),
        ("pr_number", 13, "pr_number does not match"),
        ("run_attempt", 1, "run_attempt does not match"),
        ("run_id", 140, "run_id does not match"),
        ("conclusion", "failure", "both knowledge records checks"),
        ("checks", {"kg_validate": "success"}, "both knowledge records checks"),
        (
            "checks",
            {"kg_validate": "success", "kg_repro_paths": "skipped"},
            "both knowledge records checks",
        ),
        ("schema_version", True, "schema version 1 records-only"),
        ("scope", "normal", "schema version 1 records-only"),
    ],
)
def test_rejects_invalid_or_stale_artifact(
    verify: Any,
    record: dict[str, object],
    workflow_run: dict[str, object],
    field: str,
    value: object,
    error: str,
) -> None:
    record[field] = value
    with pytest.raises(ValueError, match=error):
        verify(
            json.dumps(record).encode(),
            workflow_run,
            pr_number=12,
            head_sha=HEAD,
            base_sha=BASE,
            merge_sha=MERGE,
        )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"path": ".github/workflows/ci.yml"}, "Knowledge library pull_request run"),
        ({"event": "push"}, "Knowledge library pull_request run"),
        ({"head_sha": "d" * 40}, "workflow run HEAD"),
        (
            {"status": "in_progress", "conclusion": None},
            "workflow run has not succeeded",
        ),
        ({"conclusion": "cancelled"}, "workflow run has not succeeded"),
    ],
)
def test_rejects_unsuccessful_or_unrelated_run_even_with_a_successful_artifact(
    verify: Any,
    record: dict[str, object],
    workflow_run: dict[str, object],
    change: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        verify(
            json.dumps(record).encode(),
            {**workflow_run, **change},
            pr_number=12,
            head_sha=HEAD,
            base_sha=BASE,
            merge_sha=MERGE,
        )


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (None, "missing knowledge records result"),
        (b"{", "invalid knowledge records result JSON"),
    ],
)
def test_rejects_missing_or_unreadable_result(
    verify: Any,
    workflow_run: dict[str, object],
    data: bytes | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        verify(
            data,
            workflow_run,
            pr_number=12,
            head_sha=HEAD,
            base_sha=BASE,
            merge_sha=MERGE,
        )
