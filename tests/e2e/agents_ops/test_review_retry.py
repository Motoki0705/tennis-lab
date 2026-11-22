"""Explicit review retries through the parent CLI, controller and runner boundary."""

from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import OPS, ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_progression_cli import (
    interrupt_review,
    join_controller,
    prepared,
    progression_path,
    request,
)

SERVICE_UNAVAILABLE = "Codex backend 503 Service Unavailable"


def retry_args(
    p: ParentEnv, value: dict[str, Any], *, failed_round: int = 1
) -> tuple[str, ...]:
    return (
        "merge-continue",
        "--run-id",
        value["review_run_id"],
        "--retry-review",
        str(failed_round),
        "--instruction-file",
        p.note("一時障害後に前回と同じ対象で続行する。"),
        "--wait",
    )


@pytest.mark.parametrize("ci", ["SUCCESS", "FAILURE"])
def test_retry_after_503_obeys_current_head_ci(parent_env: ParentEnv, ci: str) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    initial = request(
        p,
        numbers,
        fake,
        REVIEW_AGENT_FINAL_EXIT="1",
        REVIEW_RUNTIME_ERROR_MESSAGE=SERVICE_UNAVAILABLE,
    )
    state_file = progression_path(p, initial)
    join_controller(state_file)
    assert initial["status"] == "parent_wait"
    assert initial["review_usage"]["rounds"][0]["error"] == SERVICE_UNAVAILABLE
    prior = state_file.parent / "review-1"
    runtime = prior / "agent-runtime/result.json"
    failed_record = runtime.read_bytes()
    assert not (prior / "result.json").exists()

    # Eligible runtime failures from before launch locks were introduced remain recoverable.
    (prior / "execution.json").write_text(
        json.dumps({"status": "running", "agent_invocations": 1})
    )
    (prior / "launch.lock").unlink()
    replay = request(p, numbers, fake)
    assert replay["review_run_id"] == initial["review_run_id"]
    ordinary = json.loads(
        p.cli("merge-continue", "--instruction-file", p.note(), "--wait").stdout
    )
    assert ordinary["status"] == "parent_wait" and ordinary["review"]["round"] == 1
    assert len(calls.read_text().splitlines()) == 1
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())
    join_controller(state_file)

    remote = p.state()
    head = remote["prs"][0]["head"]["sha"]
    remote["prs"][0]["statusCheckRollup"][0]["conclusion"] = ci
    p.save(remote)
    args = retry_args(p, initial)
    result = json.loads(p.cli(*args).stdout)
    join_controller(state_file)
    assert result["review"]["round"] == 2
    assert result["review"]["result"]["reviewed_heads"] == {str(numbers[0]): head}
    assert result["review_history"][0]["failure"]["turn_end_reason"] == "runtime_error"
    usage = json.loads(
        p.cli("merge-status", "--run-id", initial["review_run_id"]).stdout
    )["review_usage"]
    assert usage["confirmed_launches"] == 2
    assert [
        (row["round"], row["turn_end_reason"], row["error"]) for row in usage["rounds"]
    ] == [
        (1, "runtime_error", SERVICE_UNAVAILABLE),
        (2, "completed", None),
    ]
    assert runtime.read_bytes() == failed_record
    merges = [c for c in p.calls() if c[:2] == ["pr", "merge"]]
    if ci == "SUCCESS":
        assert result["status"] == "completed" and len(merges) == 1
        assert merges[0][merges[0].index("--match-head-commit") + 1] == head
    else:
        assert result["status"] == "parent_wait" and merges == []
        assert result["prs"][0]["ci"]["status"] == "failure"
    assert p.cli(*retry_args(p, result, failed_round=2), ok=False).returncode == 2
    assert len(calls.read_text().splitlines()) == 2


def test_concurrent_retry_reserves_each_round_once_and_enforces_limit(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    initial = request(p, numbers, fake, REVIEW_AGENT_FINAL_EXIT="1")
    state_file = progression_path(p, initial)
    join_controller(state_file)
    args = retry_args(p, initial)
    ready = Barrier(2, timeout=10)
    failure = {"REVIEW_AGENT_FINAL_EXIT": "1", "REVIEW_RUNTIME_REASON": "usage_limit"}

    def retry() -> subprocess.CompletedProcess[str]:
        ready.wait()
        return p.cli(*args, ok=None, extra=failure)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(retry), pool.submit(retry)
        responses = [first.result(), second.result()]
    assert sorted(response.returncode for response in responses) == [0, 2]
    paused = json.loads(
        next(response.stdout for response in responses if response.returncode == 0)
    )
    assert paused["review"]["round"] == 2 and paused["status"] == "parent_wait"
    join_controller(state_file)
    assert p.cli(*args, ok=False).returncode == 2
    assert len(calls.read_text().splitlines()) == 2

    paused = json.loads(
        p.cli(
            *retry_args(p, initial, failed_round=2),
            extra={
                "REVIEW_AGENT_FINAL_EXIT": "1",
                "REVIEW_RUNTIME_REASON": "startup_failed",
            },
        ).stdout
    )
    join_controller(state_file)
    rejected = p.cli(*retry_args(p, initial, failed_round=3), ok=False)
    assert "attempt limit reached (3)" in rejected.stderr
    assert paused["review"]["round"] == 3 and paused["status"] == "parent_wait"
    assert [row["turn_end_reason"] for row in paused["review_usage"]["rounds"]] == [
        "runtime_error",
        "usage_limit",
        "startup_failed",
    ]
    assert len(calls.read_text().splitlines()) == 3
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())


def test_retry_rejects_unconfirmed_cleanup_changed_head_and_unauthorized_parent(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    initial = request(
        p, numbers, fake, REVIEW_AGENT_FINAL_EXIT="1", REVIEW_UNCONFIRMED_CLEANUP="1"
    )
    state_file = progression_path(p, initial)
    join_controller(state_file)
    args = retry_args(p, initial)
    assert "termination or cleanup is unconfirmed" in p.cli(*args, ok=False).stderr
    runtime = state_file.parent / "review-1/agent-runtime/result.json"
    record = json.loads(runtime.read_text())
    record["cleanup_complete"] = True
    runtime.write_text(json.dumps(record))

    remote = p.state()
    head = remote["prs"][0]["head"]["sha"]
    remote["prs"][0]["head"]["sha"] = "f" * 40
    p.save(remote)
    assert "previous review target and HEAD" in p.cli(*args, ok=False).stderr
    remote["prs"][0]["head"]["sha"] = head
    p.save(remote)
    assert (
        p.cli(
            *args, ok=False, extra={"CLAUDE_CODE_SESSION_ID": "different-parent"}
        ).returncode
        == 2
    )
    p.cli("basis-revoke", "--reason-file", p.note("実行根拠を取り消す。"))
    assert p.cli(*args, ok=False).returncode == 2
    assert len(calls.read_text().splitlines()) == 1
    assert not (state_file.parent / "review-2").exists()
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())


def test_interruption_just_after_launch_intent_is_retryable(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    initial = request(p, numbers, fake, AGENT_AUTO_MERGE="0")
    state_file = progression_path(p, initial)
    join_controller(state_file)
    interrupt_review(p, state_file, when="execution.json")
    review_dir = state_file.parent / "review-1"
    execution = review_dir / "execution.json"
    assert execution.is_file() and not (review_dir / "agent-runtime").exists()
    assert not calls.read_text()
    args = retry_args(p, initial)

    # Without an inherited lock recorded in the intent, missing files prove nothing.
    guarded = execution.read_bytes()
    execution.write_text(json.dumps({"status": "running", "agent_invocations": 1}))
    assert "runtime state is unconfirmed" in p.cli(*args, ok=False).stderr
    execution.write_bytes(guarded)

    # A missing directory can still have a live git registration for a worktree.
    leftover = (
        p.repo / ".claude/worktrees" / ("review-" + initial["review_run_id"]) / "pr-100"
    )
    run_git(p.repo, "worktree", "add", "--detach", str(leftover))
    shutil.rmtree(leftover.parent)
    assert "previous review workspace remains" in p.cli(*args, ok=False).stderr
    run_git(p.repo, "worktree", "remove", "--force", str(leftover))

    result = json.loads(p.cli(*args).stdout)
    assert result["status"] == "completed"
    usage = result["review_usage"]
    assert usage["launch_intents"] == 2 and usage["confirmed_launches"] == 1
    assert usage["rounds"][0]["turn_end_reason"] == "interrupted_before_runtime"
    assert usage["rounds"][0]["agent_spawned"] is False
    assert len(calls.read_text().splitlines()) == 1


def test_live_runner_before_runtime_blocks_retry_after_controller_dies(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    initial = request(p, numbers, fake, AGENT_AUTO_MERGE="0")
    state_file = progression_path(p, initial)
    join_controller(state_file)
    ready = p.root / "runner-ready.pipe"
    os.mkfifo(ready)
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import os
import signal

with open(os.environ['REVIEW_READY_PIPE'], 'w') as receipt:
    receipt.write(str(os.getpid()))
signal.pause()
"""
    )
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    child = None
    proc = subprocess.Popen(
        [sys.executable, str(OPS / "review/merge_gate.py"), "--state", str(state_file)],
        cwd=p.repo,
        env={**p.env, "REVIEW_READY_PIPE": str(ready)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(ready_fd, selectors.EVENT_READ)
            assert selector.select(timeout=15), (
                "runner never reached its startup boundary"
            )
        child = int(os.read(ready_fd, 64))
        proc.kill()
        proc.wait(timeout=5)
        assert not (state_file.parent / "review-1/agent-runtime").exists()
        blocked = p.cli(*retry_args(p, initial), ok=False)
        assert "reviewer launcher is still active" in blocked.stderr
        assert not (state_file.parent / "review-2").exists()
        assert not calls.read_text()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if child is not None:
            os.kill(child, signal.SIGKILL)
        os.close(ready_fd)
