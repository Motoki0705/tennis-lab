"""Repair reservation replay around the actual manual launcher boundary."""

from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import (
    latest,
    manual,
    run,
    state_dir,
    store,
    task,
)
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_wait_worker import command, connected, listening, message
from tests.unit.agents_ops.test_repair_turn import report


def creating_worker(p: ParentEnv) -> Path:
    run(p)
    gh = p.state()
    gh["prs"] = [p.pr(42, 10, branch="codex/worker")]
    gh["prs"][0]["labels"] = []
    p.save(gh)
    return Path(latest(p))


def test_same_repair_reservation_before_during_and_after_launch(
    reserved: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = reserved
    first = creating_worker(p)
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(p.repo)
    api = importlib.import_module("parent.manual_workers")
    marker = importlib.import_module("lib.repair_turn").request("ci-attempt-1", 42)
    registry = store(p)
    gh = importlib.import_module("parent.github").GitHub("o/r", store=registry)
    parent_session = importlib.import_module("parent.session").ParentSession
    runner = p.root / "fake-runner"
    monkeypatch.setattr(api, "RUNNER", runner)
    prompt = Path(p.note("Repair the failed Python import."))
    with registry.operation("o/r#1"), registry.locked():
        session = parent_session(
            gh, registry, 1, "claude:" + p.env["CLAUDE_CODE_SESSION_ID"]
        )
        one = api.prepare_run(session, 10, prompt, True, None, repair=marker)
        two = api.prepare_run(session, 10, prompt, True, None, repair=marker)
        assert one == two
    assert len(task(p)["worker_runs"]) == 2
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 1
    args = (
        "run-worker",
        "--child",
        "10",
        "--resume",
        "--repair-id",
        "ci-attempt-1",
        "--repair-pr",
        "42",
        "--prompt-file",
        str(prompt),
    )
    socket_path = p.root / "repair.sock"
    # A socket handshake holds the launched runner before it creates a receipt.
    with tempfile.TemporaryDirectory(prefix="repair-wait-") as temp:
        pending = Path(temp) / "pending.sock"
        with listening(pending) as waiting_server, listening(socket_path) as listener:
            waiter = subprocess.Popen(
                command(
                    runner, "wait-worker", "--child", "10", "--after-run", str(first)
                ),
                cwd=p.repo,
                env={**p.env, "WAIT_PENDING_SOCKET": str(pending)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                with connected(waiting_server) as waiting:
                    assert message(waiting) == b"pending\n"
                    assert waiter.poll() is None and not latest(p).exists()
                    waiting.sendall(b"C")
                with ThreadPoolExecutor() as executor:
                    launched = executor.submit(
                        manual,
                        p,
                        *args,
                        runner=runner,
                        MANUAL_REPAIR_SOCKET=str(socket_path),
                        MANUAL_PLANNING_FINAL=json.dumps(report()),
                    )
                    with connected(listener) as connection:
                        assert connection.recv(7) == b"started"
                        assert waiter.poll() is None
                        duplicate = json.loads(manual(p, *args, runner=runner).stdout)
                        assert duplicate["launch_state"] == "starting"
                        assert (
                            len(
                                (p.root / "runner-calls.jsonl").read_text().splitlines()
                            )
                            == 2
                        )
                        connection.sendall(b"x")
                    result = json.loads(launched.result(timeout=45).stdout)
                stdout, stderr = waiter.communicate(timeout=15)
                assert waiter.returncode == 0, stderr + stdout
            finally:
                if waiter.poll() is None:
                    waiter.kill()
                    waiter.communicate(timeout=10)
    waited = json.loads(stdout)
    assert waited["run_dir"] == result["run_dir"]
    assert waited["wait_state"] == "ended" and waited["cleanup_complete"]
    ended = json.loads(manual(p, *args, runner=runner).stdout)
    assert ended["launch_state"] == "ended"
    assert result["run_dir"] == ended["run_dir"]
    assert len(task(p)["worker_runs"]) == 2
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 2
    assert "repair_attempts" not in p.cli("status").stdout
    current = latest(p)
    repair_prompt = (current / "prompt.md").read_text()
    assert "PARENT_RESULT" in repair_prompt
    assert "## 自動進行の修正ターン" in repair_prompt
    assert "followup" not in result
    original = json.loads((first / "result.json").read_text())
    repaired = json.loads((current / "result.json").read_text())
    assert original["runtime"]["session_id"] == repaired["runtime"]["session_id"]
    assert repaired["contract"]["settings"]["selected"]["effort"] == "high"
    assert (
        repaired["contract"]["settings"]["selected"]["model"]
        == original["contract"]["settings"]["selected"]["model"]
    )
    parse = importlib.import_module("lib.repair_turn").parse_report
    assert (
        parse(Path(ended["final_file"]).read_text(), repair_id="ci-attempt-1", pr=42)[
            "outcome"
        ]
        == "repaired"
    )
    logs = importlib.import_module("worker_logs.journal")
    schema = importlib.import_module("worker_logs.schema")
    journal = Path(task(p)["preflight"]["turns"][-1]["log"]["journal"])
    saved = logs.read(journal)
    # Fake GitHub publication may be pending, but the real public validator ran.
    assert saved["state"] in {"ready", "pending", "merged", "blocked"}
    assert saved.get("error") not in {
        "record_worker",
        "runner_settings_changed",
        "record_repair",
    }
    files = importlib.import_module("worker_logs.sources").render(
        saved["source"], saved["receipt"]
    )
    assert schema.validate(files)["worker"]["selected"]["effort"] == "high"
    run(p, "--resume")
    assert (
        json.loads((latest(p) / "result.json").read_text())["contract"]["settings"][
            "selected"
        ]["effort"]
        == "max"
    )


def test_repair_popen_rejection_reuses_reservation(reserved: ParentEnv) -> None:
    p = reserved
    creating_worker(p)
    before = len(p.calls())
    args = ("--resume", "--repair-id", "ci-2", "--repair-pr", "42")
    failed = run(p, *args, ok=False, MANUAL_ADMISSION_FAILURE="1")
    assert json.loads(failed.stdout)["launch_state"] == "rejected_before_start"
    directory = latest(p)
    assert not directory.exists()
    run(p, *args, MANUAL_PLANNING_FINAL=json.dumps({**report(), "repair_id": "ci-2"}))
    assert latest(p) == directory
    assert len(task(p)["worker_runs"]) == 2
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 2
    assert not any(
        arg.partition("?")[0] in {"repos/o/r/issues", "repos/o/r/pulls"}
        for call in p.calls()[before:]
        for arg in call
    )


def test_rejected_caller_cannot_reset_a_competing_repair_launch(
    reserved: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = reserved
    creating_worker(p)
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MANUAL_STATE_DIR", str(state_dir(p)))
    monkeypatch.setenv("MANUAL_RUNNER_CALLS", str(p.root / "runner-calls.jsonl"))
    monkeypatch.chdir(p.repo)
    api = importlib.import_module("parent.manual_workers")
    admission = importlib.import_module("parent.improvement_runtime")
    monkeypatch.setattr(api, "RUNNER", p.root / "fake-runner")
    registry = store(p)
    gh = importlib.import_module("parent.github").GitHub("o/r", store=registry)
    original = admission.execute_manual

    def rejected_after_competing_launch(*args: Any, **kwargs: Any) -> Any:
        competing = original(*args, **kwargs)
        assert competing.spawned
        # This caller was rejected before claiming. A second caller has now
        # started the same saved operation before the rejection is handled.
        return admission.ManualExecution(
            subprocess.CompletedProcess(args[5], 2, b"", b"admission rejected"), False
        )

    monkeypatch.setattr(admission, "execute_manual", rejected_after_competing_launch)
    code, result = api.run_worker(
        gh,
        registry,
        1,
        "claude:" + p.env["CLAUDE_CODE_SESSION_ID"],
        child=10,
        prompt=Path(p.note()),
        resume=True,
        restart_evidence=None,
        repair={"schema_version": 1, "repair_id": "ci-race", "pr": 42},
    )
    assert code == 0 and result["launch_state"] == "starting"
    assert task(p)["repair_attempts"]["ci-race"]["launch_claim"]
    replay = run(p, "--resume", "--repair-id", "ci-race", "--repair-pr", "42")
    assert json.loads(replay.stdout)["launch_state"] == "starting"
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 2


def test_confirmed_unstarted_repair_can_retry_only_the_same_request(
    reserved: ParentEnv,
) -> None:
    p = reserved
    first = creating_worker(p)
    args = ("--resume", "--repair-id", "ci-cancelled", "--repair-pr", "42")
    failed = json.loads(run(p, *args, ok=False, MANUAL_PREFLIGHT_FAILURE="1").stdout)
    assert failed["launch_state"] == "runtime_unconfirmed"
    directory = Path(failed["run_dir"])
    delegation = task(p)["delegation_id"]
    assert task(p)["repair_attempts"]["ci-cancelled"]["launch_state"] == "starting"
    assert not directory.exists()
    replay = json.loads(run(p, *args).stdout)
    assert replay["run_dir"] == str(directory) and replay["launch_state"] == "starting"

    result = p.cli("cancel-worker-launch", "--child", "10", "--run-dir", str(directory))
    assert json.loads(result.stdout)["log_publication"]["stage"] == "not_started"
    attempt = task(p)["repair_attempts"]["ci-cancelled"]
    assert (
        attempt["launch_state"] == "cancelled_unstarted"
        and "launch_claim" not in attempt
    )
    assert latest(p) == first
    changed = manual(
        p,
        "run-worker",
        "--child",
        "10",
        "--prompt-file",
        p.note("変更した修正指示"),
        *args,
        ok=False,
    )
    assert "already bound" in changed.stderr

    restarted = json.loads(
        run(p, *args, MANUAL_PLANNING_FINAL=json.dumps(report())).stdout
    )
    assert restarted["delegation_id"] == delegation
    assert restarted["run_dir"] != str(directory)
    assert json.loads(run(p, *args).stdout)["launch_state"] == "ended"
    calls = (p.root / "runner-calls.jsonl").read_text().splitlines()
    assert len(calls) == 3
    record = json.loads((latest(p) / "result.json").read_text())
    assert record["contract"]["repair"] == attempt["request"]
    assert (
        record["runtime"]["session_id"]
        == json.loads((first / "result.json").read_text())["runtime"]["session_id"]
    )


@pytest.mark.parametrize(
    "failure", [KeyboardInterrupt(), RuntimeError("launch response lost")]
)
def test_uncertain_popen_keeps_the_claim_and_never_resends(
    reserved: ParentEnv, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    p = reserved
    creating_worker(p)
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(p.repo)
    api = importlib.import_module("parent.manual_workers")
    admission = importlib.import_module("parent.improvement_runtime")
    runner = p.root / "fake-runner"
    monkeypatch.setattr(api, "RUNNER", runner)
    registry = store(p)
    gh = importlib.import_module("parent.github").GitHub("o/r", store=registry)
    original = admission.subprocess.Popen

    def uncertain(command: list[str], *args: Any, **kwargs: Any) -> Any:
        if command[0] == str(runner) and "--prompt-file" in command:
            raise failure
        return original(command, *args, **kwargs)

    monkeypatch.setattr(admission.subprocess, "Popen", uncertain)
    with pytest.raises(ValueError, match="repair launch is unconfirmed"):
        api.run_worker(
            gh,
            registry,
            1,
            "claude:" + p.env["CLAUDE_CODE_SESSION_ID"],
            child=10,
            prompt=Path(p.note()),
            resume=True,
            restart_evidence=None,
            repair={"schema_version": 1, "repair_id": "ci-uncertain", "pr": 42},
        )
    attempt = task(p)["repair_attempts"]["ci-uncertain"]
    assert attempt["launch_state"] == "starting"
    replay = run(p, "--resume", "--repair-id", "ci-uncertain", "--repair-pr", "42")
    assert json.loads(replay.stdout)["launch_state"] == "starting"
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("target", ["parent", "child", "pr"])
@pytest.mark.parametrize("timing", ["reserved", "claimed"])
def test_manual_repair_rechecks_needs_human_immediately_before_popen(
    reserved: ParentEnv, monkeypatch: pytest.MonkeyPatch, target: str, timing: str
) -> None:
    p = reserved
    creating_worker(p)
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MANUAL_STATE_DIR", str(state_dir(p)))
    monkeypatch.setenv("MANUAL_RUNNER_CALLS", str(p.root / "runner-calls.jsonl"))
    monkeypatch.chdir(p.repo)
    api = importlib.import_module("parent.manual_workers")
    admission = importlib.import_module("parent.improvement_runtime")
    monkeypatch.setattr(api, "RUNNER", p.root / "fake-runner")
    registry = store(p)
    gh = importlib.import_module("parent.github").GitHub("o/r", store=registry)
    original_require = admission.require
    original_save = registry.save
    primed = []

    def label_target() -> None:
        remote = p.state()
        row = (
            remote["prs"][0]
            if target == "pr"
            else remote["issues"]["1" if target == "parent" else "10"]
        )
        row["labels"].append({"name": "needs-human"})
        p.save(remote)

    def require_then_label(session: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_require(session, *args, **kwargs)
        if args == (10,):
            # Cache all three targets in this final admission transaction,
            # then change the remote labels before the Popen boundary.
            for endpoint in ("issues/1", "issues/10", "pulls/42"):
                primed.append(session.gh.get("repos/o/r/" + endpoint))
            if timing == "reserved":
                label_target()
        return result

    def save_then_label() -> None:
        original_save()
        attempt = (
            registry.parents["o/r#1"]["tasks"]["10"]
            .get("repair_attempts", {})
            .get("ci-needs-human", {})
        )
        if attempt.get("launch_state") == "starting":
            label_target()

    monkeypatch.setattr(admission, "require", require_then_label)
    if timing == "claimed":
        monkeypatch.setattr(registry, "save", save_then_label)
    code, result = api.run_worker(
        gh,
        registry,
        1,
        "claude:" + p.env["CLAUDE_CODE_SESSION_ID"],
        child=10,
        prompt=Path(p.note()),
        resume=True,
        restart_evidence=None,
        repair={"schema_version": 1, "repair_id": "ci-needs-human", "pr": 42},
    )
    assert len(primed) == 3
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 1, (
        "needs-human target reached the runner"
    )
    assert code == 2 and result["launch_state"] == "rejected_before_start"
    assert "needs-human" in result["reason"]
    assert {"parent": "parent issue #1", "child": "child issue #10", "pr": "PR #42"}[
        target
    ] in result["reason"]
    assert task(p)["repair_attempts"]["ci-needs-human"]["launch_state"] == "reserved"
    assert not latest(p).exists()
