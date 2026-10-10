"""Daily triage: checked proposals become ``ai-proposed`` child issues exactly once."""

from __future__ import annotations

import json
import re
from typing import Any


def _report_issue(ops_env: Any) -> dict[str, Any]:
    result = ops_env.run("report")
    assert result.returncode == 0, result.stderr
    issue: dict[str, Any] = next(
        i for i in ops_env.state["issues"] if "ops-report" in i["labels"]
    )
    return issue


def _edit_body(ops_env: Any, number: int, edit: Any) -> None:
    state = ops_env.state
    issue = next(i for i in state["issues"] if i["number"] == number)
    issue["body"] = edit(issue["body"])
    ops_env.write_state(state)


def _children(ops_env: Any) -> list[dict[str, Any]]:
    return [i for i in ops_env.state["issues"] if i["title"].startswith("[R-")]


def _check(body: str) -> str:
    """Check proposal 02 and option 'close' of stale-PR proposal 01."""
    body = body.replace("- [ ] **R-2026W42-02**", "- [x] **R-2026W42-02**")
    return body.replace("  - [ ] close", "  - [x] close", 1)


def test_checked_proposals_become_children_once(ops_env: Any) -> None:
    report = _report_issue(ops_env)
    _edit_body(ops_env, report["number"], _check)

    first = ops_env.run("triage")
    assert first.returncode == 0, first.stderr

    children = _children(ops_env)
    assert sorted(c["title"] for c in children) == [
        "[R-2026W42-01] #7 放置されたPR（選択: close）",
        "[R-2026W42-02] big.py を分割する",
    ]
    assert all(c["labels"] == ["ai-proposed"] for c in children)
    stale_child = next(c for c in children if c["title"].startswith("[R-2026W42-01]"))
    assert f"レポート: #{report['number']}" in stale_child["body"]
    assert "選択された対応: **close**" in stale_child["body"]
    body = next(i for i in ops_env.state["issues"] if i["number"] == report["number"])[
        "body"
    ]
    nums = {c["title"][1:13]: c["number"] for c in children}
    assert (
        f"**R-2026W42-02** big.py を分割する（優先度: 高） → #{nums['R-2026W42-02']}"
        in body
    )
    assert f"**R-2026W42-01** #7 放置されたPR → #{nums['R-2026W42-01']}" in body
    # Unchecked proposal 03 untouched.
    assert "週次レポートのプロンプトを改善する（優先度: 低）\n" in body

    second = ops_env.run("triage")
    assert second.returncode == 0, second.stderr
    assert len(_children(ops_env)) == 2
    assert json.loads(second.stdout)["created"] == []


def test_crash_between_create_and_annotate_does_not_duplicate(ops_env: Any) -> None:
    report = _report_issue(ops_env)
    _edit_body(ops_env, report["number"], _check)

    crashed = ops_env.run("triage", FAKE_GH_FAIL="issue edit")
    assert crashed.returncode == 1
    assert "injected failure" in crashed.stderr
    assert len(_children(ops_env)) == 2
    body = next(i for i in ops_env.state["issues"] if i["number"] == report["number"])[
        "body"
    ]
    assert not re.search(r"→ #\d", body)

    recovered = ops_env.run("triage")
    assert recovered.returncode == 0, recovered.stderr
    summary = json.loads(recovered.stdout)
    assert summary["created"] == []
    assert len(summary["linked_existing"]) == 2
    assert len(_children(ops_env)) == 2
    body = next(i for i in ops_env.state["issues"] if i["number"] == report["number"])[
        "body"
    ]
    assert len(re.findall(r"→ #\d", body)) == 2


def test_ambiguous_options_fail_but_other_items_proceed(ops_env: Any) -> None:
    report = _report_issue(ops_env)

    def both_options(body: str) -> str:
        body = body.replace("  - [ ] 5件に絞る", "  - [x] 5件に絞る")
        body = body.replace("  - [ ] 上限なし", "  - [x] 上限なし")
        return body.replace("- [ ] **R-2026W42-02**", "- [x] **R-2026W42-02**")

    _edit_body(ops_env, report["number"], both_options)

    result = ops_env.run("triage")

    assert result.returncode == 1
    assert "R-2026W42-03: 2 options checked" in result.stderr
    assert [c["title"] for c in _children(ops_env)] == [
        "[R-2026W42-02] big.py を分割する"
    ]


def test_dry_run_triage_does_not_write(ops_env: Any) -> None:
    report = _report_issue(ops_env)
    _edit_body(ops_env, report["number"], _check)
    before = ops_env.state

    result = ops_env.run("triage", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "would create child issue" in result.stderr
    assert ops_env.state == before


def test_closed_reports_are_not_triaged(ops_env: Any) -> None:
    report = _report_issue(ops_env)
    _edit_body(ops_env, report["number"], _check)
    state = ops_env.state
    next(i for i in state["issues"] if i["number"] == report["number"])["state"] = (
        "CLOSED"
    )
    ops_env.write_state(state)

    result = ops_env.run("triage")

    assert result.returncode == 0, result.stderr
    assert _children(ops_env) == []


def test_gh_failure_in_triage_exits_nonzero(ops_env: Any) -> None:
    result = ops_env.run("triage", FAKE_GH_FAIL="issue list")

    assert result.returncode == 1
    assert "injected failure" in result.stderr
