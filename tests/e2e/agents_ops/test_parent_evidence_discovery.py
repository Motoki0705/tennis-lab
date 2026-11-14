"""Evidence discovery cost and freshness through fake GitHub's real pagination."""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import replace

import pytest

from tests.e2e.agents_ops.test_parent_evidence import evidence_issue
from tests.e2e.agents_ops.test_parent_session import ParentEnv, ReviewEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)


def listing_calls(calls: list[list[str]]) -> list[str]:
    return [
        arg for call in calls for arg in call if arg.startswith("repos/o/r/issues?")
    ]


@pytest.mark.parametrize("command", ["status", "resolve"])
def test_commands_only_page_through_labelled_evidence(
    single_review_env: ReviewEnv, command: str
) -> None:
    r, p = single_review_env, single_review_env.p
    r.run()
    worktree = p.repo / ".claude/worktrees/pr-0"
    (worktree / "change.txt").write_text("parent verified fix\n")
    run_git(worktree, "commit", "-am", "fix checked by parent")
    head = run_git(worktree, "rev-parse", "HEAD")
    state = p.state()
    state["prs"][0]["head"]["sha"] = head
    for n in range(1000, 2000):
        state["issues"][str(n)] = {
            **copy.deepcopy(state["issues"]["1"]),
            "id": 10000 + n,
            "number": n,
            "body": "Unrelated work",
        }
    p.save(state)
    metrics_path = p.root / f"{command}-metrics.jsonl"
    extra = {"PARENT_METRICS_FILE": str(metrics_path)}
    offset = len(p.calls())
    if command == "status":
        assert json.loads(p.cli("status", extra=extra).stdout)["parent"] == 1
    else:
        # The review CLI has no metrics entrypoint; instrument only this measurement.
        driver = p.root / "review-with-metrics.py"
        driver.write_text(
            "import os, runpy\nfrom pathlib import Path\nfrom parent import metrics\n"
            "with metrics.command('review.resolve', Path(os.environ['PARENT_METRICS_FILE'])) as row:\n"
            "    try:\n"
            f"        runpy.run_path({str(r.clock_script)!r}, run_name='__main__')\n"
            "    except SystemExit as exc:\n"
            "        row['exit_code'] = exc.code\n"
            "        raise\n"
        )
        proof = p.spec(
            {
                "ci": f"Python succeeded for {head}",
                "regression_tests": "parent checked the changed line",
                "resolution": "parent verified the fix",
                "reviewed_paths": ["change.txt"],
            }
        )
        result = json.loads(
            replace(r, clock_script=driver)
            .cli(
                "resolve",
                "--run",
                r.request["review_run_id"],
                "--pr",
                "100",
                "--finding",
                "changes",
                "--fixed-head",
                head,
                "--evidence",
                proof,
                extra=extra,
            )
            .stdout
        )
        assert result["coverage"]["covered"]
    calls = p.calls()[offset:]
    (measured,) = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    api_calls = [call for call in calls if call[0] == "api"]
    assert measured["github_requests"] == len(api_calls)
    assert measured["github_pages"] == sum(
        "page=" in arg for call in api_calls for arg in call
    )
    assert measured["exit_code"] == 0
    listings = listing_calls(calls)
    report = {
        "command": command,
        "github_requests": measured["github_requests"],
        "github_pages": measured["github_pages"],
        "all_issue_pages": sum("labels=" not in endpoint for endpoint in listings),
        "evidence_pages": sum(
            "labels=parent-evidence" in endpoint for endpoint in listings
        ),
    }
    (p.root / f"{command}-measurement.json").write_text(json.dumps(report))
    print(json.dumps(report))
    assert listings
    assert all(
        "state=all&labels=parent-evidence&per_page=100&page=1" in endpoint
        for endpoint in listings
    )
    assert len(listings) == (1 if command == "status" else 3)


@pytest.mark.parametrize("duplicate", [False, True])
def test_labelled_discovery_reads_later_pages_and_rejects_duplicate_bindings(
    parent_env: ParentEnv, duplicate: bool
) -> None:
    p = parent_env
    state = p.state()
    for n in range(100, 200):
        row = evidence_issue(p, n, state="closed")
        row["body"] = row["body"].replace('"parent": 1', f'"parent": {n + 1000}')
        state["issues"][str(n)] = row
    state["issues"]["9"] = evidence_issue(p, state="closed")
    if duplicate:
        state["issues"]["10"] = evidence_issue(p, 10)
    p.save(state)
    result = p.cli("evidence", ok=not duplicate)
    if duplicate:
        assert "duplicate evidence issues" in result.stderr
    else:
        assert json.loads(result.stdout)["number"] == 9
        assert json.loads(result.stdout)["state"] == "closed"
    assert len(listing_calls(p.calls())) == 2


def test_unlabelled_marker_duplicate_is_outside_discovery(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["9"] = evidence_issue(p)
    state["issues"]["10"] = {**evidence_issue(p, 10), "labels": []}
    p.save(state)
    assert json.loads(p.cli("evidence", "--create").stdout)["number"] == 9
    assert json.loads(p.cli("evidence").stdout)["number"] == 9
    assert not any("POST" in call for call in p.calls())


@pytest.mark.parametrize("boundary", ["scope", "fresh", "write"])
def test_discovery_scope_shares_reads_and_refreshes_at_boundaries(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["9"] = evidence_issue(p)
    p.save(state)
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS", "XDG_STATE_HOME"):
        monkeypatch.setenv(key, p.env[key])
    store = importlib.import_module("parent.state").Store(p.root / "discovery-state")
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    gh.parent = 1
    manager = gh.evidence_issues()
    with store.locked():
        store.focus("o/r#1")
        with gh.history_read_scope():
            assert manager.find(1, "operator").number == 9
            assert manager.find(1, "operator").number == 9
            assert len(listing_calls(p.calls())) == 1
            state["issues"]["10"] = evidence_issue(p, 10)
            p.save(state)
            if boundary == "write":
                # An unrelated comment write must also refresh the evidence index.
                # Its endpoint does not evict the listing from the GET cache.
                gh.write(
                    "POST",
                    "repos/o/r/issues/1/comments",
                    {"body": "Parent observation recorded"},
                )
            if boundary == "fresh":
                with (
                    gh.fresh_reads(),
                    pytest.raises(ValueError, match="duplicate evidence issues"),
                ):
                    manager.find(1, "operator")
            if boundary != "scope":
                with pytest.raises(ValueError, match="duplicate evidence issues"):
                    manager.find(1, "operator")
        if boundary == "scope":
            with (
                gh.history_read_scope(),
                pytest.raises(ValueError, match="duplicate evidence issues"),
            ):
                manager.find(1, "operator")
    assert len(listing_calls(p.calls())) == 2
