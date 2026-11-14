"""Explicit session replacement through the existing manual worker entry."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import (
    issue_body,
    latest,
    manual,
    run,
    state_dir,
    store,
    task,
)
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / ".agents/ops/lib/agent_exec.sh"


def files(directory: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


def corrupt_record(p: ParentEnv) -> Path:
    """Saved post-stop evidence; the real stop path is exercised below."""
    directory: Path = latest(p)
    path = directory / "result.json"
    record = json.loads(path.read_text())
    record.update(
        result="failed",
        turn_end_reason="metadata_error",
        exit_code=1,
        recovery_error="Expecting value: line 1 column 42 (char 41)",
    )
    path.write_text(json.dumps(record))
    return directory


def replacement(p: ParentEnv, *extra: str, **kwargs: Any) -> Any:
    evidence = p.root / "restart-evidence.md"
    evidence.write_text(
        "ホスト停止後に回収済み。保存イベントの破損で同じsessionへ再開できない。"
    )
    return run(
        p,
        "--replace-session",
        "--restart-evidence-file",
        str(evidence),
        *extra,
        **kwargs,
    )


@pytest.mark.parametrize(
    "condition,diagnostic",
    [
        ("resumable", "resumable"),
        ("metadata_only", "resumable"),
        ("running", "termination"),
        ("alive", "termination"),
        ("cleanup", "cleanup"),
        ("missing_evidence", "restart evidence"),
        ("blank_evidence", "nonempty"),
        ("changed_model", "cannot change"),
        ("unknown_session_state", "session metadata"),
        ("different_latest_run", "latest session run"),
        ("contract", "runner records"),
        ("unknown_turn_source", "saved worker turn"),
        ("missing_cleanup_confirmation", "cleanup confirmation"),
        ("missing_prompt", "--prompt-file"),
        ("conflicting_mode", "not allowed"),
        ("earlier_running", "termination"),
    ],
)
def test_session_replacement_rejects_without_changing_reservation(
    reserved: ParentEnv, condition: str, diagnostic: str
) -> None:
    p = reserved
    run(p)
    earlier = latest(p)
    if condition == "earlier_running":
        run(p, "--resume")
        earlier_result = earlier / "result.json"
        data = json.loads(earlier_result.read_text())
        data["result"] = "running"
        earlier_result.write_text(json.dumps(data))
    directory = corrupt_record(p)
    path = directory / "result.json"
    record = json.loads(path.read_text())
    extra = {}
    if condition in {"resumable", "metadata_only"}:
        record.pop("recovery_error")
        if condition == "resumable":
            record.update(result="completed", turn_end_reason="completed", exit_code=0)
    elif condition == "running":
        record["result"] = "running"
    elif condition == "alive":
        extra["MANUAL_AGENT_ALIVE"] = "1"
    elif condition == "cleanup":
        record["cleanup_complete"] = False
    elif condition == "contract":
        record["contract"]["task_id"] = "f" * 32
    elif condition == "missing_cleanup_confirmation":
        record["runner_lost"] = True
    path.write_text(json.dumps(record))
    if condition == "unknown_session_state":
        (directory / "session.json").unlink()
    elif condition == "different_latest_run":
        session_file = directory / "session.json"
        saved = json.loads(session_file.read_text())
        saved["latest_run"] = str(p.root / "unregistered-newer-run")
        session_file.write_text(json.dumps(saved))
    elif condition == "unknown_turn_source":
        with store(p).locked() as registry:
            registry.parents["o/r#1"]["tasks"]["10"]["preflight"]["turns"][-1][
                "source"
            ] = str(p.root / "unrecorded-run")
            registry.save()
    before = store(p).read_data()
    old_files = files(directory)
    calls = (p.root / "runner-calls.jsonl").read_bytes()
    if condition == "missing_evidence":
        response = run(p, "--replace-session", ok=False)
    elif condition == "missing_prompt":
        response = manual(
            p,
            "run-worker",
            "--child",
            "10",
            "--replace-session",
            "--restart-evidence-file",
            p.note(),
            ok=False,
        )
    elif condition == "conflicting_mode":
        response = replacement(p, "--resume", ok=False)
    elif condition == "blank_evidence":
        blank = p.root / "blank.md"
        blank.write_text(" \n")
        response = run(
            p, "--replace-session", "--restart-evidence-file", str(blank), ok=False
        )
    else:
        response = replacement(
            p,
            *(["--model", "gpt-6-astra"] if condition == "changed_model" else []),
            ok=False,
            **extra,
        )
    assert diagnostic in response.stderr
    assert store(p).read_data() == before
    assert files(directory) == old_files
    assert (p.root / "runner-calls.jsonl").read_bytes() == calls


@pytest.mark.parametrize(
    "missing_session,legacy_phase", [(False, False), (True, False), (False, True)]
)
def test_session_replacement_preserves_history_and_binds_new_runtime(
    reserved: ParentEnv, missing_session: bool, legacy_phase: bool
) -> None:
    p = reserved
    run(p, "--model", "gpt-6-astra", "--model-reason-file", p.note("幾何の設計。"))
    directory = corrupt_record(p)
    if missing_session:
        with store(p).locked() as registry:
            phase = registry.parents["o/r#1"]["tasks"]["10"]["preflight"]
            phase["runtime"] = importlib.import_module(
                "parent.worker_preflight"
            ).runtime_binding(json.loads((directory / "result.json").read_text()))
            phase["turns"][0].pop("log")
            registry.save()
        for name in ("result.json", "session.json"):
            path = directory / name
            record = json.loads(path.read_text())
            if name == "result.json":
                record.pop("recovery_error")
                record["runtime"]["session_id"] = None
            else:
                record["session_id"] = None
            path.write_text(json.dumps(record))
    if legacy_phase:
        with store(p).locked() as registry:
            saved_task = registry.parents["o/r#1"]["tasks"]["10"]
            saved_task.pop("preflight")
            saved_task["worker_runs"][-1].pop("preflight")
            registry.save()
    before = copy.deepcopy(task(p))
    old_files = files(directory)
    replacement(p)
    saved = task(p)
    assert saved["delegation_id"] == before["delegation_id"]
    assert saved["model"] == "gpt-6-astra"
    assert saved["model_reason"] == before["model_reason"]
    assert saved["worker_runs"][:-1] == before["worker_runs"]
    assert saved["preflight"]["turns"][:-1] == before.get("preflight", {}).get(
        "turns", []
    )
    if legacy_phase:
        assert saved["preflight"]["decisions"] == []
        assert saved["preflight"]["turns"][0]["snapshot"]["previous_turn_id"] is None
    else:
        assert saved["preflight"]["lineage_id"] == before["preflight"]["lineage_id"]
    assert saved["worker_runs"][-1]["kind"] == "replacement"
    assert saved["preflight"]["turns"][-1]["log"]["state"] == "captured"
    event = saved["preflight"]["session_replacements"][-1]
    assert event["reason"] == (
        "missing_session_id" if missing_session else "recovery_error"
    )
    assert event["restart_evidence"].startswith("ホスト停止後に回収済み。")
    assert event["from_run"]["directory"] == str(directory)
    assert (
        event["from_run"]["result_sha256"]
        == hashlib.sha256(old_files["result.json"]).hexdigest()
    )
    assert event["from_run"]["termination"]["cleanup_complete"] is True
    assert event["to_run"] == str(latest(p))
    new_session = json.loads((latest(p) / "result.json").read_text())["runtime"][
        "session_id"
    ]
    assert new_session != event["from_runtime"]["session_id"]
    assert event["to_runtime"]["session_id"] == new_session
    assert saved["preflight"]["runtime"] == event["to_runtime"]
    assert files(directory) == old_files
    status = p.cli("status").stdout
    phase = json.loads(status)["tasks"]["10"]["worker_phase"]
    assert phase["session_replacement"] == {
        "count": 1,
        "state": "bound",
        "reason": event["reason"],
    }
    assert new_session not in status and str(directory) not in status
    collector = importlib.import_module("worker_status.collector")
    observed, availability = collector.latest_run(saved, {}, 10, state_dir(p), False)
    assert availability == "recorded" and observed["kind"] == "replacement"
    run(p, "--resume")
    assert task(p)["preflight"]["runtime"]["session_id"] == new_session
    old_turns = before.get("preflight", {}).get("turns", [])
    assert task(p)["preflight"]["turns"][: len(old_turns)] == old_turns


def test_rejected_session_replacement_can_be_retried_explicitly(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p)
    directory = corrupt_record(p)
    before = copy.deepcopy(task(p))
    rejected = replacement(p, ok=False, MANUAL_ADMISSION_FAILURE="1")
    assert json.loads(rejected.stdout)["launch_state"] == "rejected_before_start"
    saved = task(p)
    assert saved["worker_runs"] == before["worker_runs"]
    assert saved["preflight"]["runtime"] == before["preflight"]["runtime"]
    assert saved["preflight"]["session_replacements"][-1]["cancelled"] is True
    replacement(p)
    assert len(task(p)["worker_runs"]) == 2
    assert latest(p) != directory
    assert len(task(p)["preflight"]["session_replacements"]) == 2


@pytest.fixture
def legacy_worker(reserved: ParentEnv) -> ParentEnv:
    p = reserved
    run(p)
    corrupt_record(p)
    with store(p).locked() as registry:
        saved = registry.parents["o/r#1"]["tasks"]["10"]
        saved.pop("preflight")
        saved["worker_runs"][-1].pop("preflight")
        saved["worker_runs"][-1]["kind"] = "registered"
        registry.save()
    return p


def test_legacy_session_replacement_retries_rejected_launch_and_requests_review(
    legacy_worker: ParentEnv,
) -> None:
    p = legacy_worker
    directory = latest(p)
    old_files = files(directory)
    before = copy.deepcopy(task(p))
    tree = Path(before["worktree"])
    (tree / "feature.txt").write_text("implemented\n")
    run_git(tree, "add", "feature.txt")
    run_git(tree, "commit", "-m", "legacy worker implementation")
    gh = p.state()
    gh["prs"] = [p.pr(100, 10, run_git(tree, "rev-parse", "HEAD"), before["branch"])]
    p.save(gh)

    rejected = replacement(p, ok=False, MANUAL_ADMISSION_FAILURE="1")
    assert json.loads(rejected.stdout)["launch_state"] == "rejected_before_start"
    saved = copy.deepcopy(task(p))
    assert saved["worker_runs"] == before["worker_runs"]
    assert saved["preflight"]["runtime"] is None
    [cancelled] = saved["preflight"]["turns"]
    assert cancelled["cancelled"] is True
    [attempt] = saved["preflight"]["session_replacements"]
    assert attempt["cancelled"] is True
    assert attempt["from_run"]["directory"] == str(directory)

    registry_before = store(p).read_data()
    refused = manual(
        p,
        "review-request",
        *p.review_flags(),
        "--prs",
        "100",
        "--instruction-file",
        p.note(),
        "--evidence-file",
        p.note(),
        ok=False,
    )
    assert "no non-cancelled worker turn" in refused.stderr
    assert store(p).read_data() == registry_before

    replacement(p)
    current = task(p)
    assert current["worker_runs"][:-1] == before["worker_runs"]
    assert current["preflight"]["turns"][:-1] == saved["preflight"]["turns"]
    assert current["preflight"]["session_replacements"][:-1] == [attempt]
    assert (
        current["preflight"]["session_replacements"][-1]["from_run"]
        == attempt["from_run"]
    )
    assert files(directory) == old_files
    review = manual(
        p,
        "review-request",
        *p.review_flags(),
        "--prs",
        "100",
        "--instruction-file",
        p.note(),
        "--evidence-file",
        p.note(),
    )
    assert json.loads(review.stdout)["review_state"] == "instructed"
    phase = task(p)["preflight"]
    assert phase["accepted_turn_id"] == phase["turns"][-1]["snapshot"]["turn_id"]
    assert phase["accepted_turn_id"] != cancelled["snapshot"]["turn_id"]
    assert files(directory) == old_files


def test_legacy_session_replacement_retry_rejects_changed_source(
    legacy_worker: ParentEnv,
) -> None:
    p = legacy_worker
    directory = latest(p)
    replacement(p, ok=False, MANUAL_ADMISSION_FAILURE="1")
    result = directory / "result.json"
    record = json.loads(result.read_text())
    record["finished_at"] += 1
    result.write_text(json.dumps(record))
    before = store(p).read_data()
    calls = (p.root / "runner-calls.jsonl").read_bytes()
    refused = replacement(p, ok=False)
    assert "saved replacement source changed" in refused.stderr
    assert store(p).read_data() == before
    assert (p.root / "runner-calls.jsonl").read_bytes() == calls


@pytest.mark.parametrize("registered", [False, True])
def test_corrupt_events_stop_replace_resume_and_review_request(
    reserved: ParentEnv, registered: bool
) -> None:
    """The actual wrapper owns stop recovery and creates every runtime record."""
    p = reserved
    bin_dir = p.root / "offline-cli"
    bin_dir.mkdir()
    codex = bin_dir / "codex"
    codex.write_bytes(Path(__file__).with_name("fake_agent_cli.py").read_bytes())
    codex.chmod(0o700)
    p.env.update(
        PATH=str(bin_dir) + os.pathsep + p.env["PATH"],
        CODEX_HOME=str(p.root / "codex-home"),
        FAKE_ARGS=str(p.root / "codex-args.txt"),
        FAKE_CWD=str(p.root / "codex-cwd.txt"),
        FAKE_INVOCATIONS=str(p.root / "codex-calls.jsonl"),
    )

    def real(*options: str, ok: bool = True, session_id: str = "old-session") -> Any:
        return manual(
            p,
            "run-worker",
            "--child",
            "10",
            "--prompt-file",
            p.note("保存済み差分を確認し受入条件の検証を終える。"),
            *options,
            ok=ok,
            runner=RUNNER,
            FAKE_SESSION_ID=session_id,
        )

    real("--planning")
    initial = latest(p)
    first = json.loads((initial / "result.json").read_text())
    real("--resume")
    lost = latest(p)
    path = lost / "result.json"
    record = json.loads(path.read_text())
    record.update(
        result="running",
        turn_end_reason=None,
        exit_code=None,
        cleanup_complete=False,
        runner={"pid": 2147483647, "boot_id": "lost-test-boot", "start_ticks": "0"},
    )
    record["runtime"]["turn_id"] = None
    path.write_text(json.dumps(record))
    session_file = initial / "session.json"
    saved_session = json.loads(session_file.read_text())
    saved_session["last_turn_id"] = first["runtime"]["turn_id"]
    session_file.write_text(json.dumps(saved_session))
    with (lost / "events.jsonl").open("a") as stream:
        stream.write('{"type":')
    confirmation = p.root / "cleanup-confirmation.json"
    confirmation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": record["run_id"],
                "runner": record["runner"],
                "process": record["process"],
                "cleanup_complete": True,
                "evidence": "Offline CLI and runner exited before simulating a lost save.",
            }
        )
    )
    stopped = subprocess.run(
        [
            str(RUNNER),
            "stop",
            "--run-dir",
            str(lost),
            "--cleanup-confirmation",
            str(confirmation),
        ],
        env=p.env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert stopped.returncode == 0, stopped.stderr
    recovered = json.loads(path.read_text())
    assert recovered["cleanup_complete"] is True
    assert recovered["turn_end_reason"] == "metadata_error"
    assert "Expecting value" in recovered["recovery_error"]
    previous_runs = copy.deepcopy(task(p)["worker_runs"])
    rejected = real("--resume", ok=False)
    assert "previous turn evidence is unconfirmed" in rejected.stderr
    assert task(p)["worker_runs"] == previous_runs
    registered_path = p.root / "parent-launched-run"
    if registered:
        launched = subprocess.run(
            [
                str(RUNNER),
                "--role",
                "implementation",
                "--mode",
                "write",
                "--cwd",
                task(p)["worktree"],
                "--branch",
                task(p)["branch"],
                "--task-id",
                task(p)["delegation_id"],
                "--model",
                task(p)["model"],
                "--prompt-file",
                p.note(),
                "--run-dir",
                str(registered_path),
            ],
            env={**p.env, "FAKE_SESSION_ID": "parent-launched-session"},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert launched.returncode == 0, launched.stderr
        takeover_result = registered_path / "result.json"
        takeover = json.loads(takeover_result.read_text())
        takeover.update(
            result="failed", turn_end_reason="descendants_reaped", exit_code=1
        )
        takeover_result.write_text(json.dumps(takeover))
        manual(
            p,
            "record-worker-run",
            "--child",
            "10",
            "--run-dir",
            str(registered_path),
            runner=RUNNER,
        )
        mismatch = real("--resume", ok=False)
        assert "session/contract differs" in mismatch.stderr
    tree = Path(task(p)["worktree"])
    (tree / "feature.txt").write_text("implemented\n")
    run_git(tree, "add", "feature.txt")
    run_git(tree, "commit", "-m", "worker implementation")
    gh = p.state()
    gh["prs"] = [p.pr(100, 10, run_git(tree, "rev-parse", "HEAD"), "codex/worker")]
    p.save(gh)
    if registered:
        refused = manual(
            p,
            "review-request",
            *p.review_flags(),
            "--prs",
            "100",
            "--instruction-file",
            p.note(),
            "--evidence-file",
            p.note(),
            ok=False,
            runner=RUNNER,
        )
        assert "saved instruction" in refused.stderr
    before = copy.deepcopy(task(p))
    old_files = {
        str(d): files(d)
        for d in (initial, lost, *([registered_path] if registered else []))
    }
    evidence = p.root / "restart-evidence.md"
    evidence.write_text(
        "停止と全子孫の回収を確認。events末尾破損で元sessionは再開不能。登録済みrunも終了確認済み。"
    )
    real(
        "--replace-session",
        "--restart-evidence-file",
        str(evidence),
        session_id="replacement-session",
    )
    saved = task(p)
    assert saved["worker_runs"][:-1] == before["worker_runs"]
    assert saved["preflight"]["turns"][:-1] == before["preflight"]["turns"]
    event = saved["preflight"]["session_replacements"][-1]
    assert event["from_run"]["directory"] == str(lost)
    assert event["from_runtime"]["session_id"] == "old-session"
    assert event["to_runtime"]["session_id"] == "replacement-session"
    assert [r["directory"] for r in event["registered_runs"]] == (
        [str(registered_path)] if registered else []
    )
    real("--resume", session_id="replacement-session")
    response = manual(
        p,
        "review-request",
        *p.review_flags(),
        "--prs",
        "100",
        "--instruction-file",
        p.note(),
        "--evidence-file",
        p.note(),
        runner=RUNNER,
    )
    assert json.loads(response.stdout)["review_state"] == "instructed"
    assert (
        task(p)["preflight"]["accepted_turn_id"]
        == task(p)["preflight"]["turns"][-1]["snapshot"]["turn_id"]
    )
    assert {directory: files(Path(directory)) for directory in old_files} == old_files
    assert str(lost) not in response.stdout + json.dumps(p.state())
    assert issue_body(p, 10)
