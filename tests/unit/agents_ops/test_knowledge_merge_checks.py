"""Knowledge merge gate reads and the worker's CI lane during rollout."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.unit.agents_ops.test_knowledge_ci import BASE, HEAD, OPS
from tests.unit.agents_ops.test_knowledge_ci import record as record
from tests.unit.agents_ops.test_knowledge_ci import workflow_run as workflow_run


@pytest.fixture
def records(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("lib.ops_review.knowledge_records")


@pytest.fixture
def pr() -> dict[str, Any]:
    repo = {"full_name": "o/r", "default_branch": "main"}
    return {
        "number": 12,
        "state": "open",
        "draft": False,
        "mergeable": True,
        "labels": [],
        "head": {"sha": HEAD, "ref": "codex/records", "repo": repo},
        "base": {"sha": BASE, "ref": "main", "repo": repo},
        "merge_commit_sha": "c" * 40,
        "body": "Closes #10",
    }


def test_closing_reference_parser_includes_all_github_keywords(records: Any) -> None:
    body = (
        "Close #1, closes O/R#2, CLOSED #3\n"
        "fix #4, Fixes: o/r#5, FIXED https://github.com/O/R/issues/6\n"
        "Resolve #7, resolves <https://github.com/o/r/issues/8>, RESOLVED other/repo#9\n"
        "```text\nCloses #10\n```\n"
    )
    assert records.closing_references(body, "o/r") == {
        *(("o/r", n) for n in range(1, 9)),
        ("other/repo", 9),
    }


@pytest.mark.parametrize("second_head", [HEAD, "d" * 40])
def test_closing_issue_pages_must_describe_the_same_pr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: Any,
    pr: dict[str, Any],
    second_head: str,
) -> None:
    pr["body"] = "Closes #10\nCloses O/R#11"
    comparison = records.Comparison.from_pr(pr)

    def page(n: int, head: str, next_page: bool) -> dict[str, Any]:
        repo = {"nameWithOwner": "o/r"}
        return {
            "repository": {
                **repo,
                "pullRequest": {
                    "number": pr["number"],
                    "state": "OPEN",
                    "isDraft": False,
                    "body": pr["body"],
                    "headRefOid": head,
                    "baseRefName": "main",
                    "baseRefOid": BASE,
                    "closingIssuesReferences": {
                        "totalCount": 2,
                        "nodes": [{"number": n, "repository": repo}],
                        "pageInfo": {
                            "hasNextPage": next_page,
                            "endCursor": "next" if next_page else None,
                        },
                    },
                },
            }
        }

    graphql = Mock(side_effect=[page(10, HEAD, True), page(11, second_head, False)])
    gh = SimpleNamespace(repo="o/r", graphql=graphql)
    monkeypatch.setattr(records, "git", Mock(return_value=b""))
    if second_head == HEAD:
        assert records.closing_targets(gh, tmp_path, pr, comparison) == {10, 11}
    else:
        with pytest.raises(ValueError, match="different PR snapshot"):
            records.closing_targets(gh, tmp_path, pr, comparison)
    assert [call.args[1]["after"] for call in graphql.call_args_list] == [None, "next"]


@pytest.mark.parametrize(
    ("fault", "status", "reason"),
    [
        ("queued", "pending", "PR run is pending"),
        ("failed", "failure", "conclusion: failure"),
        ("old_base", "pending", "earlier comparison"),
        ("head", "failure", "result head_sha does not match"),
        ("test_merge", "failure", "result merge_sha does not match"),
        ("attempt", "failure", "result run_attempt does not match"),
        ("corrupt", "failure", "Expecting value"),
    ],
)
def test_actions_observation_uses_only_latest_run_and_its_attempt(
    monkeypatch: pytest.MonkeyPatch,
    records: Any,
    pr: dict[str, Any],
    record: dict[str, object],
    workflow_run: dict[str, object],
    fault: str,
    status: str,
    reason: str,
) -> None:
    newest = {**workflow_run, "run_number": 42, "head_branch": pr["head"]["ref"]}
    older = {**newest, "run_number": 41, "id": 139}
    if fault == "queued":
        newest.update(status="queued", conclusion=None)
    elif fault == "failed":
        newest.update(conclusion="failure")
    elif fault == "old_base":
        record["base_sha"] = "d" * 40
    elif fault == "head":
        record["head_sha"] = "d" * 40
    elif fault == "test_merge":
        record["merge_sha"] = "d" * 40
    elif fault == "attempt":
        record["run_attempt"] = 1
    body = b"!" if fault == "corrupt" else json.dumps(record).encode()
    download = Mock(return_value=body)
    monkeypatch.setattr(records, "result_bytes", download)
    gh = SimpleNamespace(
        repo="o/r", get=Mock(return_value={"name": "main", "commit": {"sha": BASE}})
    )
    backend = SimpleNamespace(
        repo="o/r", gh=gh, paged=Mock(return_value=[older, newest])
    )
    result = records.observe_ci(backend, pr, records.Comparison.from_pr(pr))
    assert result["status"] == status, result
    assert reason in result["reasons"][0]
    assert result["run"]["id"] == 141
    if fault in {"queued", "failed"}:
        download.assert_not_called()
    else:
        download.assert_called_once_with(backend, newest)


@pytest.mark.parametrize("fault", ["missing", "expired", "ambiguous", "corrupt_zip"])
def test_knowledge_artifact_must_be_live_unique_and_readable(
    records: Any,
    monkeypatch: pytest.MonkeyPatch,
    workflow_run: dict[str, Any],
    fault: str,
) -> None:
    artifact = {
        "name": "knowledge-records-141-2",
        "id": 25,
        "expired": fault == "expired",
        "size_in_bytes": 40,
    }
    matches = {
        "missing": [{"name": "knowledge-records-141-1"}],
        "ambiguous": [artifact, {**artifact, "id": 26}],
    }.get(fault, [artifact])
    backend = SimpleNamespace(repo="o/r", paged=Mock(return_value=matches))
    download = Mock(return_value=SimpleNamespace(returncode=0, stdout=b"not a ZIP"))
    monkeypatch.setattr(records.subprocess, "run", download)
    with pytest.raises(
        ValueError,
        match="artifact ZIP" if fault == "corrupt_zip" else "one live Actions artifact",
    ):
        records.result_bytes(backend, workflow_run)
    if fault != "corrupt_zip":
        download.assert_not_called()


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("head", "HEAD changed"),
        ("base", "main advanced"),
        ("attempt", "attempt changed"),
        ("stop", "needs-human"),
    ],
)
def test_merge_rechecks_the_last_observation_before_recording_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: Any,
    pr: dict[str, Any],
    fault: str,
    reason: str,
) -> None:
    live = importlib.import_module("lib.ops_review.progression_backend")
    store = importlib.import_module("parent.state").Store(tmp_path)
    backend = live.LiveBackend(
        {"request": {"repo": "o/r", "parent_issue": 1}, "checkout": str(tmp_path)},
        store,
    )
    target = {
        "number": 12,
        "branch": "codex/records",
        "base_ref": "main",
        "issues": [10],
    }
    run = {"id": 1, "run_attempt": 2, "status": "completed", "conclusion": "success"}
    ci = {"head": HEAD, "base_sha": BASE, "merge_sha": "c" * 40, "run": dict(run)}
    tip = BASE
    if fault == "head":
        pr["head"]["sha"] = "d" * 40
    elif fault == "base":
        tip = "d" * 40
    elif fault == "attempt":
        run.update(run_attempt=3, status="queued", conclusion=None)
    monkeypatch.setattr(backend, "pr", Mock(return_value=pr))
    monkeypatch.setattr(backend, "local", Mock(return_value=None))
    monkeypatch.setattr(
        backend, "human", Mock(return_value="needs-human" if fault == "stop" else None)
    )
    monkeypatch.setattr(records, "latest_run", Mock(return_value=run))
    monkeypatch.setattr(records, "base_tip", Mock(return_value=tip))
    monkeypatch.setattr(records, "closing_targets", Mock(return_value={10}))
    monkeypatch.setattr(live.knowledge_evidence, "require_current", Mock())
    write = Mock()
    monkeypatch.setattr(backend.gh, "write", write)
    intent = Mock()
    with pytest.raises(live.MergeHold, match=reason):
        backend.merge_knowledge(backend.state, target, ci, intent)
    write.assert_not_called()
    intent.assert_not_called()


def test_requeued_attempt_during_human_check_returns_to_ci_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: Any,
    pr: dict[str, Any],
) -> None:
    live = importlib.import_module("lib.ops_review.progression_backend")
    loop = importlib.import_module("lib.ops_review.progression")
    store = importlib.import_module("parent.state").Store(tmp_path)
    target = {
        "number": 12,
        "branch": "codex/records",
        "base_ref": "main",
        "head_sha": HEAD,
        "issues": [10],
    }
    request = {
        "schema_version": 4,
        "kind": "knowledge_merge",
        "review_run_id": "fixture",
        "repo": "o/r",
        "parent_issue": 1,
        "evidence": "fixture attestation",
        "prs": [target],
    }
    state = loop.new_state(request, owner="fixture", generation=1)
    state["checkout"] = str(tmp_path)
    path = tmp_path / "progressions" / state["id"] / "state.json"
    path.parent.mkdir(parents=True)
    loop.save(path, state)
    backend = live.LiveBackend(state, store)
    current_run = {
        "id": 1,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
    }
    monkeypatch.setattr(backend, "pr", Mock(return_value=pr))
    monkeypatch.setattr(backend, "local", Mock(return_value=None))
    monkeypatch.setattr(backend, "dependency", Mock(return_value=True))
    monkeypatch.setattr(records, "base_tip", Mock(return_value=BASE))
    monkeypatch.setattr(records, "closing_targets", Mock(return_value={10}))
    monkeypatch.setattr(live.knowledge_evidence, "require_current", Mock())
    observed_attempts = []

    def observe_run(*_args: Any) -> dict[str, Any]:
        observed_attempts.append(current_run["run_attempt"])
        return dict(current_run)

    def observe_ci(_pr: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "head": HEAD,
            "base_sha": BASE,
            "merge_sha": "c" * 40,
            "run": observe_run(),
        }

    def requeue_on_merge_check(_state: dict[str, Any], _target: dict[str, Any]) -> None:
        if observed_attempts:
            current_run.update(run_attempt=2, status="queued", conclusion=None)

    monkeypatch.setattr(backend, "human", requeue_on_merge_check)
    monkeypatch.setattr(backend, "knowledge_ci", observe_ci)
    monkeypatch.setattr(records, "latest_run", observe_run)
    write = Mock()
    monkeypatch.setattr(backend.gh, "write", write)
    engine = loop.Engine(path, backend, clock=lambda: 10)
    engine.step(engine.state["targets"]["12"])
    saved = loop.read(path)["targets"]["12"]
    assert observed_attempts == [1, 2]
    assert saved["stage"] == "ci_wait"
    assert saved["ci"]["status"] == "pending"
    assert saved["ci"]["reasons"] == ["Knowledge library run or attempt changed"]
    assert "merge_head" not in saved
    write.assert_not_called()


@pytest.mark.parametrize("python", ["success", "missing", "skipped"])
def test_worker_ci_preserves_python_and_recognizes_knowledge_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: Any,
    pr: dict[str, Any],
    python: str,
) -> None:
    wait = importlib.import_module("parent.worker_wait_sources")
    status = {"success": "success", "missing": "pending", "skipped": "failure"}[python]
    reasons = [] if python == "success" else [f"ci_python_{python}"]
    backend = SimpleNamespace(
        repo="o/r",
        pr=Mock(return_value=pr),
        ci=Mock(return_value={"status": status, "reasons": reasons}),
    )
    monkeypatch.setattr(
        records, "assess", Mock(return_value=(None, records.Comparison.from_pr(pr)))
    )
    knowledge_ci = Mock(
        return_value={
            "status": "success",
            "reasons": [],
            "workflow": "Knowledge library",
            "base_sha": BASE,
        }
    )
    monkeypatch.setattr(records, "observe_ci", knowledge_ci)
    result = wait.ci_result(
        backend, {"pr": 12, "head": HEAD}, pr["head"]["ref"], root=tmp_path
    )
    assert result["status"] == "success"
    if python == "success":
        assert "workflow" not in result
        knowledge_ci.assert_not_called()
    else:
        assert (result["workflow"], result["base_sha"]) == ("Knowledge library", BASE)
        knowledge_ci.assert_called_once()


def test_continuing_premerge_on_same_head_requires_current_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records: Any, pr: dict[str, Any]
) -> None:
    progression = importlib.import_module("lib.ops_review.knowledge_progression")
    target = {"number": 12, "stage": "ci_wait", "head_sha": HEAD}
    state = {"checkout": str(tmp_path), "targets": {"12": target}}
    backend = SimpleNamespace(
        repo="o/r", pr=Mock(return_value=pr), shape=Mock(return_value=None)
    )
    monkeypatch.setattr(
        records, "assess", Mock(return_value=(None, records.Comparison.from_pr(pr)))
    )
    with pytest.raises(ValueError, match="requires a JSON object"):
        progression.refresh_evidence(backend, state, None, f"Verified {HEAD}")
    assert "evidence_head" not in target
