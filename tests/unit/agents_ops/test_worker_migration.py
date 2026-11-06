"""Explicit legacy-result adoption needs independent, matching evidence."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "running",
        "unclean",
        "authorization",
        "requirements",
        "evidence",
        "head",
        "branch",
        "repository",
        "closed_unmerged",
        "open_with_merge_marker",
    ],
)
def test_legacy_result_proof_requires_matching_evidence(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    migration = importlib.import_module("parent.worker_migration")
    auth = {
        "generation": 1,
        "execution_basis_kind": "human_agreement",
        "execution_basis_sha256": "a" * 64,
    }
    current: dict[str, Any] = {
        "repo": "o/r",
        "child": 10,
        "authorization": auth,
        "requirements": "b" * 64,
        "branch": "codex/legacy",
    }
    task: dict[str, Any] = {
        "verification": {
            "authorization": copy.deepcopy(auth),
            "requirements_sha256": "b" * 64,
            "prs": [100],
            "evidence": "親が差分と受入条件を確認した。",
        }
    }
    record = {"result": "completed", "cleanup_complete": True, "exit_code": 0}
    pr: dict[str, Any] = {
        "number": 100,
        "body": "Closes #10",
        "state": "open",
        "merged_at": None,
        "head": {"ref": "codex/legacy", "sha": "c" * 40, "repo": {"full_name": "o/r"}},
        "base": {"ref": "main", "repo": {"full_name": "o/r"}},
    }
    reviewed = {
        "number": 100,
        "head_sha": "c" * 40,
        "branch": "codex/legacy",
        "base_ref": "main",
    }
    if change == "running":
        record["result"] = "running"
    elif change == "unclean":
        record["cleanup_complete"] = False
    elif change == "authorization":
        task["verification"]["authorization"]["generation"] = 2
    elif change == "requirements":
        task["verification"]["requirements_sha256"] = "d" * 64
    elif change == "evidence":
        task["verification"]["evidence"] = ""
    elif change == "head":
        pr["head"]["sha"] = "d" * 40
    elif change == "branch":
        pr["head"]["ref"] = "codex/other"
    elif change == "repository":
        pr["head"]["repo"]["full_name"] = "other/fork"
    elif change == "closed_unmerged":
        pr["state"] = "closed"
    elif change == "open_with_merge_marker":
        pr["merged_at"] = "2026-11-06T00:00:00Z"
        pr["merged"] = True
    if change == "none":
        migration.validate_result(task, current, record, pr, reviewed)
    else:
        with pytest.raises(ValueError, match="legacy"):
            migration.validate_result(task, current, record, pr, reviewed)


def test_legacy_approval_expires_and_preserves_same_session_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    phase = importlib.import_module("parent.worker_preflight")
    value = phase.new("old-delegation")
    value.update(
        phase="implementing",
        context={"requirements": "old"},
        runtime={"contract_sha256": "contract", "session_id": "worker"},
    )
    value["legacy_result"] = {
        "lineage_id": value["lineage_id"],
        "delegation_id": "old-delegation",
        "context_sha256": phase.digest(value["context"]),
        "runtime": copy.deepcopy(value["runtime"]),
        "invalidated": False,
    }
    assert phase.implementation_authorized(value)
    value["delegation_id"] = "continued-delegation"
    assert not phase.implementation_authorized(value)
    value["continuations"].append(
        {"from": "old-delegation", "to": "continued-delegation"}
    )
    assert phase.implementation_authorized(value)
    value["context"]["requirements"] = "changed"
    assert not phase.implementation_authorized(value)
    value["context"]["requirements"] = "old"
    phase.invalidate(value, "dependency changed")
    assert value["legacy_result"]["invalidated"] is True
    assert not phase.implementation_authorized(value)
