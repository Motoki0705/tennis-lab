"""Capture-only recovery, runner causality and explicit missing observations."""

from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import graph

C = importlib.import_module("decision_logs.capture")
OBS = importlib.import_module("decision_logs.observations")
PARENT = importlib.import_module("decision_logs.parent_commands")


def records(op: Any) -> list[dict[str, Any]]:
    return [
        e["record"] for e in op.journal.read()["entries"] if e["event"] == "capture"
    ]


def test_journal_failure_is_visible_and_replay_never_reexecutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []
    original = C.Journal.capture

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("injected journal write failure")

    monkeypatch.setattr(C.Journal, "capture", broken)
    op = C.Operation.begin(tmp_path, "o/r", 1, "parent.run-worker", "start_ai")
    token = C.CURRENT.set(op)
    try:
        calls.append("original operation")
        assert C.observe({"exit_code": 17}, status="failure")["status"] == "incomplete"
    finally:
        C.CURRENT.reset(token)
    assert "capture incomplete" in capsys.readouterr().err
    assert C.status(tmp_path, "o/r", 1)["status"] == "incomplete"
    frozen = json.loads(op.path.read_text())["entries"]
    monkeypatch.setattr(C.Journal, "capture", original)
    view = C.status(tmp_path, "o/r", 1, retry=True)
    assert view["status"] == "captured" and not view["failures"]
    assert calls == ["original operation"]
    assert records(op) == [e["record"] for e in frozen]
    assert len(records(op)) == 3
    journal_bytes = op.journal.path.read_bytes()
    assert C.status(tmp_path, "o/r", 1, retry=True)["status"] == "captured"
    assert op.journal.path.read_bytes() == journal_bytes
    assert stat.S_IMODE(op.path.stat().st_mode) == 0o600


def test_outbox_failure_has_an_independent_missing_capture_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = C.atomic

    def fail_outbox(path: Path, value: Any) -> None:
        if path.parent.name == "captures":
            raise OSError("outbox unavailable")
        original(path, value)

    monkeypatch.setattr(C, "atomic", fail_outbox)
    C.Operation.begin(tmp_path, "o/r", 1, "parent.stop", "stop")
    view = C.status(tmp_path, "o/r", 1)
    assert view["status"] == "incomplete"
    assert view["failures"] == [
        {"source": "parent.stop", "error": "capture_delivery_missing"}
    ]


def test_runner_start_end_and_missing_final_are_not_parent_confirmation(
    tmp_path: Path,
) -> None:
    op = C.Operation.begin(
        tmp_path, "o/r", 1, "parent.run-worker", "start_ai", issue=10
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    C.runtime_environment(runtime, operation=op)
    value = {
        "result": "running",
        "agent_spawned": True,
        "run_id": "private-run",
        "runtime": {"session_id": "private-session"},
        "started_at": 1000.0,
    }
    OBS.runtime_record(runtime / "result.json", value, operation=op)
    value.update(
        result="failed", turn_end_reason="usage_limit", exit_code=1, finished_at=1002.0
    )
    OBS.runtime_record(runtime / "result.json", value, operation=op)
    OBS.runtime_record(runtime / "result.json", value, operation=op)
    rows = records(op)
    assert [r["data"]["state"] for r in rows if r["kind"] == "action"] == [
        "requested",
        "running",
        "ended",
    ]
    results = [r for r in rows if r["kind"] == "result"]
    assert [r["data"]["status"] for r in results] == ["unknown", "failure"]
    assert "usage_limit" in results[-1]["data"]["summary"]
    assert results[-1]["data"]["final_report"] == {
        "status": "unavailable",
        "reason": "missing_final",
    }
    assert all(
        s["parent_confirmation"] == "unrecorded" for s in graph.states(rows).values()
    )
    assert C.status(tmp_path, "o/r", 1)["status"] == "captured"


def test_adapter_failure_remains_detectable_before_any_result_can_be_built(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = C.Operation.begin(tmp_path, "o/r", 1, "parent.run-worker", "start_ai")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    C.runtime_environment(runtime, operation=op)
    original = OBS.runtime_auxiliary

    def fail(*args: Any) -> Any:
        raise ValueError("injected adapter failure")

    monkeypatch.setattr(OBS, "runtime_auxiliary", fail)
    value = {
        "result": "completed",
        "agent_spawned": True,
        "runtime": {},
        "exit_code": 0,
    }
    OBS.runtime_record(runtime / "result.json", value, operation=op)
    assert C.status(tmp_path, "o/r", 1)["status"] == "incomplete"
    monkeypatch.setattr(OBS, "runtime_auxiliary", original)
    OBS.runtime_record(runtime / "result.json", value, operation=op)
    assert C.status(tmp_path, "o/r", 1)["status"] == "captured"


def test_resume_request_that_actually_starts_an_ai_retains_both_facts(
    tmp_path: Path,
) -> None:
    op = C.Operation.begin(tmp_path, "o/r", 1, "parent.continue-worker", "continue_ai")
    env = C.runtime_environment(
        tmp_path / "runtime", operation=op, actual_operation="start_ai"
    )
    actual = C.Operation.load(Path(env[C.RUNTIME_ENV]))
    rows = records(op)
    assert [r["data"]["operation"] for r in rows if r["kind"] == "action"] == [
        "continue_ai",
        "continue_ai",
        "start_ai",
    ]
    assert actual.row["previous_result_id"] == next(
        r["id"] for r in rows if r["kind"] == "result"
    )
    assert graph.validate_graph(rows)


@pytest.mark.parametrize(
    "corruption", ["journal_missing", "outbox_private", "source_changed"]
)
def test_missing_media_or_changed_receipt_is_never_reported_as_captured(
    tmp_path: Path, corruption: str
) -> None:
    op = C.Operation.begin(tmp_path, "o/r", 1, "parent.note", "record")
    if corruption == "journal_missing":
        op.journal.path.unlink()
    else:
        value = json.loads(op.path.read_bytes())
        if corruption == "outbox_private":
            value["entries"][0]["private"] = {"sessions": []}
        else:
            value["entries"][0]["record"]["data"]["summary"] = "Changed frozen request"
        op.path.write_text(json.dumps(value))
    assert C.status(tmp_path, "o/r", 1)["status"] == "incomplete"


def test_two_processes_reload_one_shared_outbox_before_appending(
    tmp_path: Path,
) -> None:
    op = C.Operation.begin(tmp_path, "o/r", 1, "parent.note", "record")
    assert C.__file__ is not None
    ops_directory = str(Path(C.__file__).parents[1])
    script = """
from pathlib import Path
import sys
from decision_logs.capture import Operation
op = Operation.load(Path(sys.argv[1]))
print('loaded', flush=True)
sys.stdin.readline()
with op.transaction():
    op.result({'writer': sys.argv[2]}, artifact='concurrent source', status='success', summary='writer-' + sys.argv[2])
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(op.path), str(i)],
            env={**os.environ, "PYTHONPATH": ops_directory},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in (1, 2)
    ]
    try:
        for process in processes:
            assert (
                process.stdout is not None
                and process.stdout.readline().strip() == "loaded"
            )
        for process in processes:
            assert process.stdin is not None
            process.stdin.write("append\n")
            process.stdin.flush()
        for process in processes:
            _, error = process.communicate(timeout=10)
            assert process.returncode == 0, error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    rows = records(op)
    results = [r for r in rows if r["kind"] == "result"]
    assert {r["data"]["summary"] for r in results} == {"writer-1", "writer-2"}
    assert results[1]["data"]["previous_result_id"] == results[0]["id"]
    assert C.status(tmp_path, "o/r", 1)["status"] == "captured"


def test_explicit_decision_retains_spec_and_ids_across_journal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = C.Operation.begin(tmp_path, "o/r", 1, "parent.inspect-worker", "fetch")
    with base.transaction():
        base.finish({"observed": True})
    spec = {
        "inputs": [base.records("result")[-1]["id"]],
        "evaluation": "investigate",
        "reason": "Inspect the remaining boundary",
        "recheck_scope": ["unverified boundary"],
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(spec))
    session = SimpleNamespace(
        owner=lambda: {},
        store=SimpleNamespace(directory=tmp_path),
        gh=SimpleNamespace(repo="o/r"),
        parent=1,
        session="codex:private-parent",
    )
    original = C.Journal.capture

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("injected journal failure")

    monkeypatch.setattr(C.Journal, "capture", fail)
    with pytest.raises(ValueError, match="journal_unavailable"):
        PARENT.decide(session, path)
    frozen = [
        entry
        for receipt in base.path.parent.glob("*.json")
        for entry in C.Operation.load(receipt).row["entries"]
        if entry["record"]["kind"] == "decision"
    ]
    assert len(frozen) == 1 and frozen[0]["observation"]["spec"] == spec
    monkeypatch.setattr(C.Journal, "capture", original)
    assert C.status(tmp_path, "o/r", 1, retry=True)["status"] == "captured"
    replayed = PARENT.decide(session, path)
    assert replayed["decision_id"] == frozen[0]["record"]["id"]
    assert len([r for r in records(base) if r["kind"] == "decision"]) == 1


def test_known_environment_credentials_are_private_in_capture_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTURE_TEST_API_KEY", "opaque-credential-for-capture")
    op = C.Operation.begin(tmp_path, "o/r", 1, "parent.note", "record")
    with op.transaction():
        op.finish(
            {"observed": True},
            summary="Saved opaque-credential-for-capture in private evidence",
        )
    assert "opaque-credential-for-capture" in json.dumps(records(op))
    value = C.status(tmp_path, "o/r", 1)
    assert value["status"] == "captured"
    assert "opaque-credential-for-capture" not in json.dumps(value)
    assert "[private identifier]" in json.dumps(value)
