"""Facts remain distinct from meaning, and cross-run leases fail closed."""

from __future__ import annotations

import copy
import importlib
import json
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


@pytest.mark.parametrize(
    "state", ["running", "published", "publication_failed", "unknown"]
)
def test_unconfirmed_dead_owner_does_not_release_another_run(
    modules: Any, tmp_path: Path, state: str
) -> None:
    _, locks = modules
    a, b = manifest("a"), manifest("b")
    execution = tmp_path / "reviews" / a["review_run_id"] / "execution.json"
    execution.parent.mkdir(parents=True)
    execution.write_text(
        json.dumps(
            {"status": state, "agent_invocations": 1, "request_sha256": locks.digest(a)}
        )
    )
    reviews = [{"request": a}, {"request": b}]
    if state in {"running", "unknown"}:
        with pytest.raises(ValueError):
            locks.require_finished_overlaps(tmp_path, reviews, b)
    else:
        locks.require_finished_overlaps(tmp_path, reviews, b)
    # A separate PR must not wait for this owner, even if it is interrupted.
    locks.require_finished_overlaps(tmp_path, reviews, manifest("c", 101))


def test_pr_lease_releases_after_exception_without_unlinking_inode(
    modules: Any, tmp_path: Path
) -> None:
    _, locks = modules
    a, b = manifest("a"), manifest("b")
    with locks.target_locks(tmp_path, a):
        with pytest.raises(ValueError, match="active"), locks.target_locks(tmp_path, b):
            pytest.fail("overlapping lease was acquired")
        with locks.target_locks(tmp_path, manifest("c", 101)):
            pass
    files = {p: p.stat().st_ino for p in (tmp_path / "review-targets").iterdir()}
    with pytest.raises(RuntimeError), locks.target_locks(tmp_path, b):
        raise RuntimeError("operator interrupted")
    with locks.target_locks(tmp_path, a):
        assert {p: p.stat().st_ino for p in files} == files


def test_missing_followup_receipt_does_not_read_unverified_git_objects(
    modules: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = importlib.import_module("parent_review_policy")
    helper = importlib.import_module("review_evidence")
    request = manifest("a")
    record = {
        "schema_version": 1,
        "kind": "evidence",
        "review_run_id": request["review_run_id"],
        "result_sha256": "d" * 64,
        "parent_issue": 1,
        "pr": 100,
        "wait": 1,
        "snapshot": request["prs"][0],
        "evidence": {
            "ci": "same HEAD success",
            "regression_tests": "fixed input passed",
            "resolution": "parent confirmed",
            "reviewed_paths": ["change.txt"],
        },
    }
    monkeypatch.setattr(
        helper,
        "changed_paths",
        lambda *_: pytest.fail("unpublished follow-up accessed git"),
    )
    comment = {"id": 1, "user": {"login": "operator"}, "body": policy.marker(record)}
    result = helper.followups(
        tmp_path,
        {"review_run_id": request["review_run_id"]},
        [comment],
        {},
        "operator",
        {},
        {},
    )
    assert result == [{"record": record, "errors": ["local_receipt_missing"]}]
