"""Read-only CLI observations use real git and the existing fake review/publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.e2e.agents_ops.test_parent_session import ReviewEnv, run_git
from tests.e2e.agents_ops.test_parent_session import pair_review_env as pair_review_env
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.e2e.agents_ops.test_review_recovery import (
    next_request,
    recover,
    recovery_request,
)


def files(directory: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


def inspect(r: ReviewEnv, *prs: int, **extra: str) -> tuple[Any, dict[str, Any]]:
    # Viewing does not require owning the parent or having a native session.
    result = r.p.cli(
        "review-inspect",
        "--prs",
        *map(str, prs),
        ok=None,
        extra={"CODEX_SESSION_ID": "", "CLAUDE_CODE_SESSION_ID": "", **extra},
    )
    assert result.stdout, result.stderr
    return result, json.loads(result.stdout)


def test_inspect_before_prepare_and_published_review_never_mutates_state(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    r.run()
    state_dir = r.run_dir.parents[1]
    before, github, calls, invocations = (
        files(state_dir),
        r.p.state(),
        len(r.p.calls()),
        len(r.calls()),
    )
    response, value = inspect(r, 100)
    assert response.returncode == 0, response.stdout + response.stderr
    pr = value["prs"][0]
    run = pr["history"][0]
    assert pr["latest_request"] == r.request["review_run_id"]
    assert run["result"]["coverage"] == "reviewed" and run["publication_errors"] == []
    assert run["coverage_check"]["covered"] is True
    assert run["comparison"]["changed_paths"] == []
    assert all(field["equal"] for field in run["comparison"]["fields"].values())
    assert run["input_changes"]["pr:100"]["comments"] == {
        "status": "compared",
        "added": [],
        "deleted": [],
        "edited": [],
    }
    assert (
        r.p.state() == github
        and files(state_dir) == before
        and len(r.calls()) == invocations
    )
    assert all(
        c[0] == "api" and c[c.index("--method") + 1] == "GET"
        for c in r.p.calls()[calls:]
    )
    assert (
        "recommended_action" not in json.dumps(value)
        and value["agent_invocations"] == 0
    )
    # No review exists for this valid PR. Viewing must not create its first request.
    registry_path = state_dir / "registry.json"
    registry = json.loads(registry_path.read_text())
    registry["parents"]["o/r#1"]["reviews"] = []
    registry_path.write_text(json.dumps(registry))
    before = files(state_dir)
    response, value = inspect(r, 100)
    assert response.returncode == 0
    assert (
        value["prs"][0]["latest_request"] is None and value["prs"][0]["history"] == []
    )
    assert files(state_dir) == before and len(r.calls()) == invocations


def test_inspect_input_changes_and_legacy_missing_comment_baseline(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    state = r.p.state()
    state["comments"]["10"] = [
        {"id": 7000, "user": {"login": "human"}, "body": "original requirement"}
    ]
    state["pull_comments"] = {
        "100": [{"id": 8000, "user": {"login": "human"}, "body": "inline before"}]
    }
    r.p.save(state)
    r.run()
    saved = json.loads((r.run_dir / "context.json").read_text())["review_observations"][
        "items"
    ]
    assert saved["issue:10"]["comments"][0]["body"] == "original requirement"
    assert saved["pr:100"]["comments"][0]["kind"] == "inline"
    state = r.p.state()
    state["issues"]["10"]["body"] += "\nA changed acceptance requirement\n"
    state["comments"]["10"][0]["body"] = "edited requirement"
    state["pull_comments"]["100"][0]["body"] = "inline after"
    state["prs"][0]["body"] += "\nAn editorial note\n"
    r.p.save(state)
    response, value = inspect(r, 100)
    assert response.returncode == 0, response.stdout
    run = value["prs"][0]["history"][0]
    assert run["input_changes"]["issue:10"]["requirements"]["changed"] is True
    assert run["input_changes"]["issue:10"]["comments"]["edited"][0]["id"] == 7000
    assert run["input_changes"]["pr:100"]["body"]["changed"] is True
    assert run["input_changes"]["pr:100"]["comments"]["edited"][0]["id"] == 8000
    assert (
        run["coverage_check"]["covered"] is True
    )  # no meaning is assigned to prose changes
    path = r.run_dir / "context.json"
    context = json.loads(path.read_text())
    del context["review_observations"]
    path.write_text(json.dumps(context))
    response, value = inspect(r, 100)
    assert response.returncode == 0
    changes = value["prs"][0]["history"][0]["input_changes"]
    assert changes["issue:10"]["comments"]["status"] == "baseline_unavailable"
    assert changes["pr:100"]["comments"]["status"] == "baseline_unavailable"


def test_inspect_keeps_other_pr_facts_on_retrieval_failure_and_detects_race(
    pair_review_env: ReviewEnv,
) -> None:
    r = pair_review_env
    r.run()
    response, value = inspect(r, 100, 101, PARENT_FAIL_GET_PATH="repos/o/r/pulls/100")
    assert response.returncode == 2
    a, b = value["prs"]
    assert a["errors"][0]["scope"] == "pr"
    assert b["current_snapshot"]["number"] == 101
    assert b["history"][0]["result"]["coverage"] == "reviewed"
    response, value = inspect(
        r,
        100,
        101,
        PARENT_FAIL_GET_PATH="repos/o/r/issues/100/comments?per_page=100&page=1",
    )
    assert response.returncode == 2
    b_view = value["prs"][1]["history"][0]
    assert b_view["publication_unknown_prs"] == [100]
    assert (
        b_view["projection_errors"] == [] and b_view["result"]["coverage"] == "reviewed"
    )
    state = r.p.state()
    original_body = state["prs"][0]["body"]
    state["prs"][0]["body"] = {"malformed": "body"}
    r.p.save(state)
    response, value = inspect(r, 100, 101)
    assert (
        response.returncode == 2
        and value["prs"][0]["errors"][0]["error_type"] == "ValueError"
    )
    assert value["prs"][1]["current_snapshot"]["number"] == 101
    state["prs"][0]["body"] = original_body
    r.p.save(state)
    response, value = inspect(
        r, 100, PARENT_READ_HEAD_RACE=r.request["prs"][0]["base_sha"]
    )
    assert response.returncode == 2
    assert any(
        e["error_type"] == "ChangedDuringInspection" for e in value["prs"][0]["errors"]
    )


def test_inspect_preserves_cross_findings_and_shows_resolve(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    output = json.loads(r.output.read_text())
    output["verdict"] = "request_changes"
    output["cross_findings"] = [
        {
            "prs": [100],
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "contract",
            "detail": "fix the boundary",
        }
    ]
    r.output.write_text(json.dumps(output))
    r.run()
    _, value = inspect(r, 100)
    assert value["prs"][0]["history"][0]["coverage_check"]["unresolved"] == [
        [100, "cross-1"]
    ]
    worktree = r.p.repo / ".claude/worktrees/pr-0"
    (worktree / "change.txt").write_text("boundary fixed\n")
    run_git(worktree, "commit", "-am", "fix boundary")
    fixed = run_git(worktree, "rev-parse", "HEAD")
    state = r.p.state()
    state["prs"][0]["head"]["sha"] = fixed
    r.p.save(state)
    proof = r.p.root / "resolution.json"
    proof.write_text(
        json.dumps(
            {
                "ci": "same HEAD success",
                "regression_tests": "boundary checked",
                "resolution": "parent inspected the whole fix",
                "reviewed_paths": ["change.txt"],
            }
        )
    )
    r.cli(
        "resolve",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "cross-1",
        "--fixed-head",
        fixed,
        "--evidence",
        str(proof),
    )
    response, value = inspect(r, 100)
    assert response.returncode == 0, response.stdout
    run = value["prs"][0]["history"][0]
    assert run["result"]["findings"][0]["id"] == "cross-1"
    assert run["coverage_check"] == {"covered": True, "reasons": [], "unresolved": []}
    assert (
        run["followups"][0]["snapshot_matches"] and run["followups"][0]["errors"] == []
    )
    assert len(r.calls()) == 1


def test_inspect_shows_latest_failure_candidate_and_recovery_instead_of_old_approve(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    r.run()
    b = next_request(r)
    response, pending = inspect(r, 100)
    assert response.returncode == 0
    assert pending["prs"][0]["latest_request"] == b.request["review_run_id"]
    assert pending["prs"][0]["history"][-1]["result"] is None
    assert pending["prs"][0]["history"][-1]["execution"] is None
    output = json.loads(b.output.read_text())
    output["verdict"] = "request_changes"
    output["prs"][0]["findings"] = [
        {
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "new concern",
            "detail": "retained after runtime failure",
        }
    ]
    b.output.write_text(json.dumps(output))
    b.run(ok=False, extra={"REVIEW_AGENT_FINAL_EXIT": "3", "REVIEW_FINAL_ONLY": "1"})
    response, value = inspect(r, 100)
    assert response.returncode == 0, response.stdout
    pr = value["prs"][0]
    assert pr["latest_request"] == b.request["review_run_id"]
    a_view, b_view = pr["history"]
    assert a_view["result"]["verdict"] == "approve"
    assert a_view["subsequent_requests"] == [b.request["review_run_id"]]
    assert (
        b_view["result"]["status"] == "failed"
        and b_view["result"]["coverage"] == "not_reviewed"
    )
    assert b_view["saved_output"]["status"] == "candidate_only"
    assert b_view["saved_output"]["prs"][0]["findings"][0]["severity"] == "major"
    adopted = recovery_request(b)
    recover(b, adopted)
    response, value = inspect(r, 100)
    assert response.returncode == 0, response.stdout
    latest = value["prs"][0]["history"][-1]
    assert (
        latest["result"]["kind"] == "adoption"
        and latest["result"]["source_run"] == b.request["review_run_id"]
    )
    assert latest["execution"]["agent_invocations"] == 0
    assert latest["coverage_check"]["unresolved"] == [[100, "pr-100-1"]]
    assert len(r.calls()) == 2
