"""Evidence closure through the issue event and maintenance CLI."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e.agents_ops.test_parent_evidence import evidence_issue
from tests.e2e.agents_ops.test_parent_session import ParentEnv, evidence_number
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

WORKFLOW = (
    Path(__file__).resolve().parents[3] / ".github/workflows/parent-evidence-close.yml"
)


def _evidence(p: ParentEnv, issue: int, parent: int, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = evidence_issue(p, issue)
    row["body"] = row["body"].replace('"parent": 1', f'"parent": {parent}')
    row.update(fields)
    return row


def _run_close_workflow(
    p: ParentEnv, parent: int, *, reveal_on_retry: bool = False
) -> subprocess.CompletedProcess[str]:
    config = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert "closed" in config["on"]["issues"]["types"]
    assert config["permissions"]["issues"] == "write"
    step = config["jobs"]["close"]["steps"][-1]
    assert step["env"]["PARENT_ISSUE_NUMBER"] == "${{ github.event.issue.number }}"
    # Reuse installed dependencies and advance the fake listing on retry.
    setup = """
uv() {
    while [[ "$1" != python ]]; do shift; done
    shift
    "$TEST_PYTHON" "$@"
}
sleep() {
    if [[ "$REVEAL_ON_RETRY" == 1 ]]; then
        "$TEST_PYTHON" - "$PARENT_GH_STATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text())
state.pop("unlisted_evidence", None)
path.write_text(json.dumps(state))
PY
    fi
}
"""
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", setup + step["run"]],
        cwd=p.repo,
        env={
            **p.env,
            "CLAUDE_CODE_SESSION_ID": "",
            "GITHUB_REPOSITORY": "o/r",
            "PARENT_ISSUE_NUMBER": str(parent),
            "REVEAL_ON_RETRY": str(int(reveal_on_retry)),
            "TEST_PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )


@pytest.mark.parametrize("via_merge", [False, True], ids=["manual", "pr-merge"])
def test_issue_closed_workflow_preserves_comment_and_history(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, via_merge: bool
) -> None:
    p = parent_env
    p.cli("register")
    note_text = "閉じた証拠用Issueから参照する記録。"
    note = json.loads(p.cli("note", "--body-file", p.note(note_text)).stdout)
    state = p.state()
    number = evidence_number(state)
    # A different closed parent must be left for the bulk maintenance command.
    state["issues"]["2"]["state"] = "closed"
    state["issues"]["8"] = _evidence(p, 8, 2)
    p.save(state)
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS", "XDG_STATE_HOME"):
        monkeypatch.setenv(key, p.env[key])
    endpoint = "repos/o/r/issues/comments/" + note["url"].split("#issuecomment-")[1]
    with closing(importlib.import_module("parent.github").GitHub("o/r")) as gh:
        before = gh.get(endpoint)
        assert before["html_url"] == note["url"]
        if via_merge:
            pr = p.pr(100, 1)
            state["prs"] = [pr]
            p.save(state)
            subprocess.run(
                [
                    "gh",
                    "pr",
                    "merge",
                    "100",
                    "--merge",
                    "--match-head-commit",
                    pr["head"]["sha"],
                ],
                cwd=p.repo,
                env=p.env,
                check=True,
                capture_output=True,
                timeout=10,
            )
            assert p.state()["prs"][0]["merged"] is True
        else:
            gh.write(
                "PATCH",
                "repos/o/r/issues/1",
                {"state": "closed", "state_reason": "completed"},
            )

        assert p.state()["issues"]["1"]["state"] == "closed"
        offset = len(p.calls())
        run = _run_close_workflow(p, parent=1)
        assert run.returncode == 0, run.stdout + run.stderr
        result = json.loads(run.stdout)
        assert result["dry_run"] is False
        assert [(row["evidence"], row["status"]) for row in result["results"]] == [
            (number, "closed")
        ]
        assert not any("user" in call for call in p.calls()[offset:])
        after = p.state()
        assert after["issues"][str(number)]["state"] == "closed"
        assert after["issues"]["8"]["state"] == "open"
        assert gh.get(endpoint) == before

        # A stale listing may still report open. The CLI should use the
        # binding from the direct read and still return the original history.
        after["issue_listing_overrides"] = {str(number): {"state": "open"}}
        p.save(after)
        assert json.loads(p.cli("evidence").stdout)["state"] == "closed"
        p.cli("status")
        record = next(
            r for r in gh.record_history(1, "operator").records if r.url == note["url"]
        )
        assert record.body == before["body"]
        assert record.payload["body_sha256"] == note["body_sha256"]
        del after["issue_listing_overrides"]
        p.save(after)
        retry = _run_close_workflow(p, parent=1)
        assert retry.returncode == 0, retry.stdout + retry.stderr
        assert json.loads(retry.stdout)["results"][0]["status"] == "already_closed"
        assert (
            sum(
                "PATCH" in call and f"repos/o/r/issues/{number}" in call
                for call in p.calls()
            )
            == 1
        )


def test_bulk_dry_run_and_apply_use_only_verified_closed_parent_bindings(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    issues = state["issues"]
    issues["1"].update(
        state="closed",
        closed_at="2026-10-31T12:00:00Z",
        labels=[{"name": "issue:standalone"}],
    )
    issues["3"] = {**issues["1"], "id": 10003, "number": 3}
    issues["4"] = {
        **issues["3"],
        "id": 10004,
        "number": 4,
        "labels": [{"name": "issue:child"}],
    }
    issues["8"] = _evidence(p, 8, 1, created_at="2026-11-01T00:00:00Z")
    issues["9"] = _evidence(p, 9, 2)
    issues["10"] = _evidence(p, 10, 3)
    issues["11"] = _evidence(p, 11, 3, state="closed")
    issues["12"] = _evidence(p, 12, 1, body="Missing marker")
    issues["13"] = _evidence(p, 13, 1, user={"login": "other"})
    issues["14"] = _evidence(p, 14, 1, pull_request={"url": "pr"})
    issues["15"] = _evidence(p, 15, 1, labels=[{"name": "issue:child"}])
    issues["16"] = _evidence(p, 16, 4)
    p.save(state)

    dry_run = json.loads(p.cli("evidence-close", parent=None).stdout)
    assert dry_run["dry_run"] is True
    assert {row["evidence"]: row["status"] for row in dry_run["results"]} == {
        8: "would_close",
        9: "parent_open",
        10: "duplicate_binding",
        12: "unverified_binding",
        13: "unverified_binding",
        14: "unverified_binding",
        16: "would_close",
    }
    assert dry_run == json.loads(
        p.cli("evidence-close", "--dry-run", parent=None).stdout
    )
    assert p.state() == state
    assert not any("PATCH" in call for call in p.calls())

    interrupted = p.cli(
        "evidence-close",
        "--apply",
        parent=None,
        extra={"PARENT_FAIL_PATCH_ISSUE": "8"},
        ok=False,
    )
    assert interrupted.returncode == 2
    applied_statuses = {
        row["evidence"]: row["status"]
        for row in json.loads(interrupted.stdout)["results"]
    }
    assert applied_statuses[8] == "error"
    assert applied_statuses[16] == "closed"
    issues["16"]["state"] = "closed"
    assert p.state() == state

    applied = json.loads(p.cli("evidence-close", "--apply", parent=None).stdout)
    assert [
        row["evidence"] for row in applied["results"] if row["status"] == "closed"
    ] == [8]
    issues["8"]["state"] = "closed"
    assert p.state() == state
    applied_again = json.loads(p.cli("evidence-close", "--apply", parent=None).stdout)
    assert all(row["status"] != "closed" for row in applied_again["results"])
    assert p.state() == state
    assert [call[-3] for call in p.calls() if "PATCH" in call] == [
        "repos/o/r/issues/8",
        "repos/o/r/issues/16",
        "repos/o/r/issues/8",
    ]


def test_close_rejects_a_binding_changed_since_listing(parent_env: ParentEnv) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"]["state"] = "closed"
    state["issues"]["9"] = _evidence(p, 9, 2)
    state["issue_listing_overrides"] = {"9": {"body": _evidence(p, 9, 1)["body"]}}
    p.save(state)
    result = p.cli("evidence-close", "--apply", ok=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["results"] == [
        {
            "evidence": 9,
            "parent": 1,
            "status": "error",
            "reason": "evidence issue binding changed since listing",
        }
    ]
    assert p.state() == state
    assert not any("PATCH" in call for call in p.calls())


def test_scoped_close_reports_missing_binding(parent_env: ParentEnv) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"]["state"] = "closed"
    p.save(state)

    result = p.cli("evidence-close", "--apply", ok=None)
    assert result.returncode == 3, result.stdout + result.stderr
    assert json.loads(result.stdout)["results"] == [
        {"parent": 1, "status": "binding_not_found"}
    ]
    run = _run_close_workflow(p, parent=1)
    assert run.returncode == 3, run.stdout + run.stderr
    assert run.stdout.count("binding_not_found") == 3
    assert p.state() == state
    assert not any("PATCH" in call for call in p.calls())


def test_issue_closed_workflow_retries_after_lagged_listing(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"]["state"] = "closed"
    state["issues"]["8"] = _evidence(p, 8, 1)
    state["unlisted_evidence"] = [8]
    p.save(state)

    run = _run_close_workflow(p, parent=1, reveal_on_retry=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "binding_listing_mismatch" in run.stdout
    assert p.state()["issues"]["8"]["state"] == "closed"
    assert [call[-3] for call in p.calls() if "PATCH" in call] == ["repos/o/r/issues/8"]


def test_partial_listing_cannot_hide_duplicate_binding(parent_env: ParentEnv) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"]["state"] = "closed"
    state["issues"]["8"] = _evidence(p, 8, 1)
    state["issues"]["9"] = _evidence(p, 9, 1, state="closed")
    state["unlisted_evidence"] = [9]
    p.save(state)

    for scope in ("1", None):
        result = p.cli("evidence-close", "--apply", parent=scope, ok=None)
        assert result.returncode == 3, result.stdout + result.stderr
        assert json.loads(result.stdout)["results"] == [
            {
                "parent": 1,
                "status": "binding_listing_mismatch",
                "labelled_evidence": [8],
                "repository_evidence": [8, 9],
            }
        ]
        assert p.state() == state
    state["unlisted_evidence"] = []
    p.save(state)
    confirmed = p.cli("evidence-close", "--apply", ok=None)
    assert confirmed.returncode == 2, confirmed.stdout + confirmed.stderr
    assert json.loads(confirmed.stdout)["results"][0]["status"] == "duplicate_binding"
    assert p.state() == state
    assert not any("PATCH" in call for call in p.calls())
