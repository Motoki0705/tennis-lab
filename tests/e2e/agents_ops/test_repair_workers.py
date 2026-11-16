"""Repair reservation replay around the actual manual launcher boundary."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
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
from tests.unit.agents_ops.test_repair_turn import report


def creating_worker(p: ParentEnv) -> Path:
    run(p)
    gh = p.state()
    gh["prs"] = [p.pr(42, 10, branch="codex/worker")]
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
    with socket.socket(socket.AF_UNIX) as listener, ThreadPoolExecutor() as executor:
        listener.settimeout(30)
        listener.bind(str(socket_path))
        listener.listen()
        launched = executor.submit(
            manual,
            p,
            *args,
            runner=runner,
            MANUAL_REPAIR_SOCKET=str(socket_path),
            MANUAL_PLANNING_FINAL=json.dumps(report()),
        )
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(30)
            assert connection.recv(7) == b"started"
            duplicate = json.loads(manual(p, *args, runner=runner).stdout)
            assert duplicate["launch_state"] == "starting"
            assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 2
            connection.sendall(b"x")
        result = json.loads(launched.result(timeout=45).stdout)
    ended = json.loads(manual(p, *args, runner=runner).stdout)
    assert ended["launch_state"] == "ended"
    assert result["run_dir"] == ended["run_dir"]
    assert len(task(p)["worker_runs"]) == 2
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 2
    assert "repair_attempts" not in p.cli("status").stdout
    current = latest(p)
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
    args = ("--resume", "--repair-id", "ci-2", "--repair-pr", "42")
    failed = run(p, *args, ok=False, MANUAL_ADMISSION_FAILURE="1")
    assert json.loads(failed.stdout)["launch_state"] == "rejected_before_start"
    directory = latest(p)
    assert not directory.exists()
    run(p, *args, MANUAL_PLANNING_FINAL=json.dumps({**report(), "repair_id": "ci-2"}))
    assert latest(p) == directory
    assert len(task(p)["worker_runs"]) == 2
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 2


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
