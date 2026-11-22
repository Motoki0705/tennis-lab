"""Facts remain distinct from meaning, and cross-run leases fail closed."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> Any:
    for path in (OPS, OPS / "review", OPS / "lib"):
        monkeypatch.syspath_prepend(str(path))
    return importlib.import_module("review_observations"), importlib.import_module(
        "review_locks"
    )


@pytest.mark.parametrize(
    "change",
    [
        "same",
        "progress",
        "prose",
        "requirement",
        "comments",
        "missing",
        "missing_comments",
    ],
)
def test_observation_reports_changes_without_classifying_meaning(
    modules: Any, change: str
) -> None:
    observations, _ = modules
    before: Any = {
        "kind": "issue",
        "number": 12,
        "title": "Acceptance",
        "body": "## Purpose\nKeep the boundary.\n\n## 進捗概要\nBefore\n",
        "comments": [
            {"kind": "discussion", "id": 1, "login": "human", "body": "A"},
            {"kind": "discussion", "id": 2, "login": "human", "body": "B"},
        ],
    }
    after = copy.deepcopy(before)
    if change == "progress":
        after["body"] = after["body"].replace("Before", "After")
    elif change == "prose":
        after["body"] = after["body"].replace(
            "Keep the boundary.", "Keep this boundary."
        )
    elif change == "requirement":
        after["body"] = after["body"].replace(
            "Keep the boundary.", "Reject missing evidence."
        )
    elif change == "comments":
        after["comments"] = [
            {"kind": "discussion", "id": 1, "login": "human", "body": "edited"},
            {"kind": "inline", "id": 1, "login": "human", "body": "new"},
        ]
    elif change == "missing":
        before = None
    elif change == "missing_comments":
        before["comments"] = None
    result = observations.compare_observation(before, after)
    if change == "missing":
        assert result == {"status": "baseline_unavailable"}
    else:
        assert result["body"]["changed"] is (
            change in {"progress", "prose", "requirement"}
        )
        assert result["requirements"]["changed"] is (change in {"prose", "requirement"})
        if change == "missing_comments":
            assert result["comments"] == {"status": "baseline_unavailable"}
        elif change == "comments":
            assert [(c["kind"], c["id"]) for c in result["comments"]["added"]] == [
                ("inline", 1)
            ]
            assert [c["id"] for c in result["comments"]["deleted"]] == [2]
            assert result["comments"]["edited"][0]["after"]["body"] == "edited"
    assert "recommended_action" not in result and "irrelevant" not in result


def manifest(run: str, pr: int = 100) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review_run_id": "parent-1-" + run * 32,
        "repo": "o/r",
        "parent_issue": 1,
        "reason": "explicit request",
        "prs": [
            {
                "number": pr,
                "head_sha": "a" * 40,
                "base_sha": "b" * 40,
                "base_ref": "main",
                "merge_base_sha": "b" * 40,
                "diff_sha256": "c" * 64,
                "branch": "codex/test",
            }
        ],
    }


@pytest.mark.parametrize("version", [0, 2, True])
def test_unknown_observation_schema_is_not_an_empty_baseline(
    modules: Any, version: Any
) -> None:
    observations, _ = modules
    with pytest.raises(ValueError, match="observation version"):
        observations.baseline(
            {"review_observations": {"schema_version": version, "items": {}}}, None
        )
