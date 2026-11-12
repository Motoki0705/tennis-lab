"""Review capture media failures preserve the canonical operation and stay visible."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from tests.e2e.agents_ops.test_decision_capture import captured, sourced
from tests.e2e.agents_ops.test_manual_worker_runs import state_dir
from tests.e2e.agents_ops.test_parent_session import ReviewEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.e2e.agents_ops.test_review_recovery import inject_lost_response


def decision(r: ReviewEnv, command: str) -> list[str]:
    source = sourced(captured(state_dir(r.p)), "parent.review-request")[-1]
    value = json.loads(
        r.p.cli(
            "decide",
            "--spec",
            r.p.spec(
                {
                    "inputs": [source["id"]],
                    "evaluation": "investigate",
                    "reason": "Use the existing fixed review request",
                    "next_actions": [
                        {
                            "operation": "review"
                            if command == "run"
                            else "retry_publication",
                            "summary": "Observe the fixed review operation",
                            "actor": "external",
                            "target": "PR 100",
                            "instruction": "Use the saved request",
                            "conditions": "Pinned inputs",
                            "expected_result": "Canonical review response",
                        }
                    ],
                }
            ),
        ).stdout
    )
    if command == "run":
        r.p.cli(
            "review-request",
            "--run-id",
            r.request["review_run_id"],
            "--instruction-file",
            r.p.note("固定した要求を実行する。"),
            "--decision-id",
            value["decision_id"],
        )
    return ["--decision-id", value["decision_id"]]


def inject(r: ReviewEnv, fault: str) -> str:
    assert r.clock_script is not None
    original: str = r.clock_script.read_text()
    injected = """
from contextlib import contextmanager
from decision_logs import capture as capture_module
from worker_logs.journal import PublicationError
saved_lock = capture_module.locked
@contextmanager
def broken_lock(path):
    if path.parent.name == 'captures':
        raise OSError('injected review capture outbox lock failure')
    with saved_lock(path):
        yield
"""
    if fault == "outbox_lock":
        injected += "capture_module.locked = broken_lock\n"
    elif fault == "outbox_save":
        injected += """
def broken_save(self):
    raise OSError('injected review capture outbox save failure')
capture_module.Operation._save = broken_save
"""
    else:
        injected += """
def broken_journal_read(self):
    raise PublicationError('journal_unavailable')
capture_module.Journal.read = broken_journal_read
"""
    # The fixture's clock still runs the real script after this test-only injection.
    r.clock_script.write_text(
        original.replace("runpy.run_path(", injected + "\nrunpy.run_path(")
    )
    return original


@pytest.mark.parametrize(
    ("command", "with_decision", "fault", "original_fails"),
    [
        ("run", True, "outbox_lock", False),
        ("run", False, "outbox_lock", True),
        ("run", False, "outbox_save", False),
        ("resume-publication", True, "journal_read", False),
        ("resume-publication", False, "outbox_lock", True),
    ],
)
def test_capture_initialization_failure_preserves_review_and_missing_diagnostic(
    single_review_env: ReviewEnv,
    command: str,
    with_decision: bool,
    fault: str,
    original_fails: bool,
) -> None:
    r = single_review_env
    immutable = None
    failure_path = ""
    if command != "run":
        inject_lost_response(r, target="repos/o/r/issues/100/comments", method="POST")
        r.run(ok=False)
        immutable = (r.run_dir / "result.json").read_bytes()
        pending = next(c for c in r.p.comments() if '"kind":"pending"' in c["body"])
        failure_path = f"repos/o/r/issues/comments/{pending['id']}"
    flags = decision(r, command) if with_decision else []
    original_clock = inject(r, fault)
    extra = (
        {"REVIEW_AGENT_FINAL_EXIT": "3"}
        if command == "run" and original_fails
        else {"PARENT_FAIL_PATH": failure_path}
        if original_fails
        else {}
    )
    args = (
        ["run", "--request", str(r.request_path), "--agent-exec", str(r.fake)]
        if command == "run"
        else [command, "--run", r.request["review_run_id"]]
    )
    response = r.cli(*args, *flags, ok=not original_fails, extra=extra)
    assert response.returncode == (1 if original_fails else 0)
    value = json.loads(response.stdout)
    assert value["result"]["status"] == (
        "failed" if command == "run" and original_fails else "succeeded"
    )
    assert "capture incomplete" in response.stderr
    assert len(r.calls()) == 1
    if immutable is not None:
        assert (r.run_dir / "result.json").read_bytes() == immutable
    # Remove only the injected media fault. Delivery recovery must still expose
    # facts lost before capture, and must never invoke a reviewer or publication.
    assert r.clock_script is not None
    r.clock_script.write_text(original_clock)
    comments = r.p.state()["comments"]
    view = json.loads(r.p.cli("decision-log-status", ok=False).stdout)
    assert view["status"] == "incomplete" and view["failures"]
    retry = json.loads(r.p.cli("retry-decision-capture", ok=False).stdout)
    assert retry["status"] == "incomplete" and retry["failures"]
    assert len(r.calls()) == 1 and r.p.state()["comments"] == comments


@pytest.mark.parametrize("command", ["run", "resume-publication"])
def test_invalid_decision_is_still_rejected_before_review_operation(
    single_review_env: ReviewEnv, command: str
) -> None:
    r = single_review_env
    if command != "run":
        r.run()
    before = len(r.calls())
    comments = r.p.state()["comments"]
    args = (
        ["run", "--request", str(r.request_path), "--agent-exec", str(r.fake)]
        if command == "run"
        else [command, "--run", r.request["review_run_id"]]
    )
    response = r.cli(*args, "--decision-id", uuid4().hex, ok=False)
    assert response.returncode == 2 and "invalid_log_reference" in response.stderr
    assert len(r.calls()) == before and r.p.state()["comments"] == comments
