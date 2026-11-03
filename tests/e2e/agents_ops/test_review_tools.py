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
