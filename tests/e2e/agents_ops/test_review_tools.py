"""Formal reviews select pinned revisions without inheriting an audit ledger."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_ops_report_retrieval import (
    Scenario,
)
from tests.e2e.agents_ops.test_ops_report_retrieval import (
    scenario as scenario,
)

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path.insert(0, str(OPS))
server: Any = importlib.import_module("lib.review_tools.review_server")


def test_multiple_pinned_commits_exclude_dirty_worktree_and_other_revisions(
    scenario: Scenario,
) -> None:
    s = scenario
    client = server.ReviewRetrieval(
        {
            "repo": str(s.repo),
            "commits": [s.base, s.later],
            "repository": "o/r",
            "run_id": "review-1",
            "journal_dir": str(s.journal / "formal"),
        }
    )
    for sha, expected in [(s.base, "needle = 1\n"), (s.later, "newer code\n")]:
        value = client.call(
            "code_file",
            {
                "revision": sha,
                "path": "model.py",
                "start_line": 1,
                "line_count": 1,
            },
        )
        assert value["ok"], value
        assert value["data"]["text"] == expected
        assert value["sources"][0]["sha"] == sha
        assert value["base_sha"] == sha
    for revision in ["main", "f" * 40]:
        value = client.call("code_file", {"revision": revision, "path": "model.py"})
        assert not value["ok"] and value["error"]["code"] == "INVALID_ARGUMENT"
    assert not client.call("code_file", {"path": "model.py"})["ok"]
    assert not client.call("github_write", {})["ok"]
    status = client.call("cpu_test_status", {"revision": s.base})
    assert status["ok"] and not status["data"]["enabled"]
    assert "cpu_tests" not in client.tools
    records = [
        json.loads(line) for line in client.journal.path.read_text().splitlines()
    ]
    assert any(row.get("response", {}).get("ok") for row in records)


def test_cpu_allowlist_cannot_name_unpinned_revision(scenario: Scenario) -> None:
    s = scenario
    with pytest.raises(ValueError, match="unpinned"):
        server.ReviewRetrieval(
            {
                "repo": str(s.repo),
                "commits": [s.base],
                "repository": "o/r",
                "run_id": "review-1",
                "journal_dir": str(s.journal / "formal"),
                "cpu_tests": {s.later: ["tests/test_x.py"]},
            }
        )


def test_formal_review_ci_log_uses_shared_sanitization(scenario: Scenario) -> None:
    scenario.fail("controlled_log", log="\x1b[31mfailed\x1b[0m\r\n")
    client = server.ReviewRetrieval(
        {
            "repo": str(scenario.repo),
            "commits": [scenario.base],
            "repository": "o/r",
            "run_id": "review-controls",
            "journal_dir": str(scenario.journal / "formal"),
        }
    )
    response = client.call("ci_log", {"job_id": 9})
    assert response["ok"], response
    assert response["data"]["text"] == "failed\\u000d\n"
    assert response["sources"][1]["text_sanitization"]["ansi_sequences_removed"] == 2
    record = json.loads(client.journal.path.read_text().splitlines()[-1])
    assert record["response"] == response


def test_failed_ci_fetch_then_fetch_only_retry_is_captured(
    scenario: Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = importlib.import_module("decision_logs.capture")
    owner = capture.Operation.begin(
        scenario.journal / "parent-state", "o/r", 1, "review.run", "review"
    )
    monkeypatch.setenv(capture.RETRIEVAL_ENV, str(owner.path))
    client = server.ReviewRetrieval(
        {
            "repo": str(scenario.repo),
            "commits": [scenario.base],
            "repository": "o/r",
            "run_id": "review-ci",
            "journal_dir": str(scenario.journal / "formal"),
        }
    )
    scenario.fail("403")
    first = client.call("ci_runs", {"head_sha": scenario.base})
    assert not first["ok"]
    scenario.fail("")
    second = client.call("ci_runs", {"head_sha": scenario.base})
    assert second["ok"], second
    value = capture.status(scenario.journal / "parent-state", "o/r", 1)
    assert value["status"] == "captured", value
    rows = [entry["record"] for entry in value["records"]]
    assert [
        r["data"]["operation"]
        for r in rows
        if r["kind"] == "action" and r["data"]["previous_action_id"] is None
    ] == ["review", "fetch", "retry_fetch"]
    results = [r for r in rows if r["kind"] == "result"]
    assert [r["data"]["status"] for r in results] == ["failure", "success"]
    assert results[1]["data"]["previous_result_id"] == results[0]["id"]
    assert results[1]["data"]["changes"]["instruction"] == "unchanged"
    assert results[1]["data"]["changes"]["conditions"] == "unconfirmed"
    assert all(
        r["data"]["auxiliary"]["usage"]["status"] == "unavailable" for r in results
    )
    assert len(scenario.calls()) == 2
