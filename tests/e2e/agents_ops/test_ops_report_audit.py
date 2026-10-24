"""Discovery-driven rounds, native evidence, partial publication and same-issue resume."""

from __future__ import annotations

import importlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report"))
audit: Any = importlib.import_module("opsreport.audit")
agent: Any = importlib.import_module("opsreport.agent")
native: Any = importlib.import_module("opsreport.native")


def _reports(ops: Any) -> list[dict[str, Any]]:
    return [r for r in ops.state["issues"] if "ops-report" in r["labels"]]


def _state_path(ops: Any, name: str) -> Path:
    return Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report" / name


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _latest(ops: Any) -> Path:
    paths: list[Path] = ops.run_dirs()
    return max(paths, key=lambda p: p.stat().st_mtime_ns)


def _tasks(run: Path) -> dict[str, Any]:
    tasks: dict[str, Any] = _json(run / "audit.json")["ledger"]["tasks"]
    return tasks


def test_large_unrequested_lists_and_logs_never_enter_launch_input(
    ops_env: Any,
) -> None:
    first = ops_env.run("report", "--dry-run")
    assert first.returncode == 0, first.stderr
    small_prompt = (_latest(ops_env) / "prompt.md").read_text()
    state = ops_env.state
    state["prs"] *= 5000
    state["runs"] = [{"log": "PRIVATE_UNREQUESTED_LOG" * 100000}] * 100
    state["merged_prs"] *= 5000
    ops_env.write_state(state)
    second = ops_env.run("report", "--dry-run")
    assert second.returncode == 0, second.stderr
    run = _latest(ops_env)
    large_prompt = (run / "prompt.md").read_text()
    assert large_prompt == small_prompt
    assert all(
        key not in large_prompt
        for key in (
            "COLLECTED_JSON",
            '"audit_plan"',
            "PRIVATE_UNREQUESTED_LOG",
            "big.py",
            "largest_code_files",
        )
    )
    pinned = _json(run / "collected.json")
    assert set(pinned) == {
        "meta",
        "ops_memory",
        "previous_reports",
        "audit_input_sha256",
    }
    config = _json(run / "mcp.json")["ops_report"]["args"]
    journal = Path(config[config.index("--journal-dir") + 1]) / "retrieval.jsonl"
    calls = [
        _json_row
        for _json_row in map(json.loads, journal.read_text().splitlines())
        if _json_row["event"] == "requested"
    ]
    ranges = [r["arguments"] for r in calls if r["tool"] == "code_file"]
    assert ranges and all(r["start_line"] == 2 and r["line_count"] == 3 for r in ranges)
    assert not any(c[:2] in (["pr", "list"], ["run", "list"]) for c in ops_env.calls())


def test_first_result_changes_followups_and_continues_before_independent_finishes(
    ops_env: Any,
) -> None:
    first = ops_env.run("report", "--dry-run", FAKE_AUDIT_DISCOVERY="dynamic")
    assert first.returncode == 0, first.stderr
    run = _latest(ops_env)
    tasks = _tasks(run)
    assert len(tasks) > 4
    assert "refs_dynamic" in tasks
    assert tasks["refs_dynamic"]["origin"] == {
        "kind": "finding",
        "id": "initial_finding",
    }
    evidence = native.inspect_run(run / "agent-runtime")
    names = {a.task_name: a for a in evidence.agents}
    assert (
        names["/root/initial"].completed_at
        < names["/root/refs_dynamic"].started_at
        < names["/root/independent"].completed_at
    )
    assert len(
        native.reconcile(
            run / "agent-runtime", {k: t["agent_id"] for k, t in tasks.items()}
        )
    ) == len(tasks)
    second = ops_env.run("report", "--dry-run", FAKE_AUDIT_DISCOVERY="dependency")
    assert second.returncode == 0, second.stderr
    changed = _tasks(_latest(ops_env))
    assert "refs_dependency" in changed and "refs_dynamic" not in changed
    assert tasks["initial"]["conclusion"] != changed["initial"]["conclusion"]


def test_thread_limit_rejection_is_recorded_then_retried_after_completion(
    ops_env: Any,
) -> None:
    result = ops_env.run("report", FAKE_AUDIT_REJECT_ONCE="1")
    assert result.returncode == 0, result.stderr
    run = _latest(ops_env)
    observed = native.inspect_run(run / "agent-runtime")
    assert len(observed.rejections) == 1
    assert observed.rejections[0].task_name == "capacity"
    capacity = _tasks(run)["capacity"]
    assert capacity["status"] == "completed"
    assert sum(a.task_name == "/root/capacity" for a in observed.agents) == 1
    checkpoint = _json(_state_path(ops_env, "audit/20261012.json"))
    events = _json(Path(checkpoint["ledger"]))["events"]
    assert any(
        e["kind"] == "task.recorded"
        and e["data"]["id"] == "capacity"
        and e["data"]["status"] == "failed"
        and e["data"]["agent_id"] is None
        for e in events
    )


def test_unavailable_spawn_is_incomplete_and_never_uses_independent_cli(
    ops_env: Any,
) -> None:
    result = ops_env.run("report", FAKE_AUDIT_SPAWN_UNAVAILABLE="1")
    assert result.returncode == 1, result.stderr
    assert "status=incomplete" in _reports(ops_env)[0]["body"]
    assert "native起動0件" in _reports(ops_env)[0]["body"]
    assert not native.inspect_run(_latest(ops_env) / "agent-runtime").agents
    assert not _state_path(ops_env, "memory.json").exists()


def test_partial_resumes_same_issue_and_reuses_completed_tasks(ops_env: Any) -> None:
    failed = ops_env.run("report", FAKE_AUDIT_FAIL="independent")
    assert failed.returncode == 1, failed.stderr
    partial = _reports(ops_env)[0]
    assert "status=incomplete" in partial["body"]
    first_run = _latest(ops_env)
    first_tasks = _tasks(first_run)
    pinned = _json(first_run / "collected.json")
    (ops_env.repo / "big.py").write_text("new main content\n")
    subprocess.run(["git", "add", "big.py"], cwd=ops_env.repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "advance main",
        ],
        cwd=ops_env.repo,
        check=True,
    )
    recovered = ops_env.run("report", "--if-due", "--now", "2026-10-21T08:00:00+09:00")
    assert recovered.returncode == 0, recovered.stderr
    assert (
        len(_reports(ops_env)) == 1
        and _reports(ops_env)[0]["number"] == partial["number"]
    )
    assert "status=complete" in _reports(ops_env)[0]["body"]
    run = _latest(ops_env)
    assert _json(run / "collected.json") == pinned
    assert [a.task_name for a in native.inspect_run(run / "agent-runtime").agents] == [
        "/root/independent"
    ]
    assert _tasks(run)["initial"] == first_tasks["initial"]
    checkpoint = _json(_state_path(ops_env, "audit/20261012.json"))
    assert checkpoint["version"] == 2 and len(checkpoint["attempts"]) == 2
    assert (
        _json(Path(checkpoint["ledger"]))["events"][-1]["data"]["status"] == "completed"
    )


@pytest.mark.parametrize(
    "problem",
    ["checkpoint", "ledger", "hash", "closed", "selection", "base", "version1"],
)
def test_resume_inconsistency_stops_before_agent_or_publication(
    ops_env: Any, problem: str
) -> None:
    assert ops_env.run("report", FAKE_AUDIT_FAIL="independent").returncode == 1
    path = _state_path(ops_env, "audit/20261012.json")
    checkpoint = _json(path)
    state = ops_env.state
    if problem == "checkpoint":
        path.unlink()
    elif problem == "ledger":
        Path(checkpoint["ledger"]).unlink()
    elif problem == "hash":
        checkpoint["collected"]["meta"]["period_start"] = "2020-01-01T00:00:00+00:00"
        path.write_text(json.dumps(checkpoint))
    elif problem == "version1":
        path.write_text(json.dumps({"version": 1, "collected": {}, "results": {}}))
    elif problem == "closed":
        next(r for r in state["issues"] if "ops-report" in r["labels"])["state"] = (
            "CLOSED"
        )
    elif problem == "selection":
        row = next(r for r in state["issues"] if "ops-report" in r["labels"])
        row["body"] = row["body"].replace(
            "- [ ] **R-20261012-01**", "- [x] **R-20261012-01**"
        )
    elif problem == "base":
        row = next(r for r in state["issues"] if "ops-report" in r["labels"])
        row["body"] = row["body"].replace(
            f"base={checkpoint['collected']['meta']['base_sha']}", "base=" + "b" * 40
        )
    ops_env.write_state(state)
    before = ops_env.state
    Path(ops_env.env["FAKE_AGENT_USED"]).unlink()
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 1, result.stderr
    assert not Path(ops_env.env["FAKE_AGENT_USED"]).exists()
    assert ops_env.state == before
    if problem == "version1":
        assert "#1458" in result.stderr and "version 1" in result.stderr


def test_missing_receipt_is_not_publishable_and_can_be_restored_in_same_round(
    ops_env: Any,
) -> None:
    assert ops_env.run("report", FAKE_AUDIT_FAIL="independent").returncode == 1
    checkpoint = _json(_state_path(ops_env, "audit/20261012.json"))
    ledger = _json(Path(checkpoint["ledger"]))
    evidence = next(
        e["data"]
        for e in reversed(ledger["events"])
        if e["kind"] == "evidence.recorded"
    )
    receipt = evidence["reference"].rsplit(" / receipt=", 1)[1]
    journal = _state_path(ops_env, "audit/20261012/sources/retrieval.jsonl")
    rows = [json.loads(r) for r in journal.read_text().splitlines()]
    missing = next(
        r for r in rows if r["event"] == "received" and r["receipt"] == receipt
    )
    journal.write_text(
        "".join(
            json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n"
            for r in rows
            if r is not missing
        )
    )
    bad = ops_env.run("report", "--if-due")
    assert bad.returncode == 1, bad.stderr
    assert "根拠の取得記録が欠落" in _reports(ops_env)[0]["body"]
    assert "status=incomplete" in _reports(ops_env)[0]["body"]
    with journal.open("a") as out:
        out.write(json.dumps(missing, separators=(",", ":"), ensure_ascii=False) + "\n")
    recovered = ops_env.run("report", "--if-due")
    assert recovered.returncode == 0, recovered.stderr
    assert len(_reports(ops_env)) == 1
    assert "status=complete" in _reports(ops_env)[0]["body"]


def test_incomplete_report_is_not_triaged(ops_env: Any) -> None:
    assert ops_env.run("report", FAKE_AUDIT_FAIL="independent").returncode == 1
    before = ops_env.state
    result = ops_env.run("triage")
    assert result.returncode == 0, result.stderr
    assert ops_env.state == before


def test_no_whole_audit_timeout_and_mcp_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.write_text("result")
    calls = []
    monkeypatch.setattr(
        agent.shell, "run_process_group", lambda cmd, **kw: calls.append((cmd, kw))
    )
    agent.execute(
        agent.AgentRun(
            "codex",
            tmp_path / "exec",
            tmp_path,
            tmp_path / "prompt",
            out,
            tmp_path,
            tmp_path / "mcp.json",
        )
    )
    assert calls[0][1]["timeout"] is None
    cmd = calls[0][0]
    assert cmd[cmd.index("--mcp-config") + 1] == str(tmp_path / "mcp.json")


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_parent_interrupt_reaps_processes_and_resumes_same_issue(
    ops_env: Any, stop_signal: int
) -> None:
    pid_file = ops_env.tmp / "child.pid"
    proc = ops_env.start("report", FAKE_AGENT_HANG="1", FAKE_AGENT_SPAWN=str(pid_file))
    path = _state_path(ops_env, "audit/20261012.ledger.json")
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if path.exists():
                events = _json(path)["events"]
                if (
                    sum(
                        e["kind"] == "task.recorded"
                        and e["data"]["status"] == "running"
                        for e in events
                    )
                    >= 2
                ):
                    break
            assert proc.poll() is None, proc.communicate()
            time.sleep(0.05)
        else:
            raise AssertionError("fake children did not start")
        proc.send_signal(stop_signal)
        _, err = proc.communicate(timeout=20)
        assert proc.returncode == 1, err
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    child = Path(f"/proc/{int(pid_file.read_text())}/stat")
    assert not child.exists() or child.read_text().split(") ", 1)[1][0] == "Z"
    assert "status=incomplete" in _reports(ops_env)[0]["body"]
    recovered = ops_env.run("report", "--if-due")
    assert recovered.returncode == 0, recovered.stderr
    assert len(_reports(ops_env)) == 1
    assert "status=complete" in _reports(ops_env)[0]["body"]


def test_dirty_audit_checkout_is_preserved_for_explicit_repair(ops_env: Any) -> None:
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ops_env.repo, text=True
    ).strip()
    path = audit.worktree_path(ops_env.repo, "test-dirty")
    with (
        pytest.raises(audit.shell.CommandError, match="contains modified or untracked"),
        audit.checkout(ops_env.repo, base, path) as snapshot,
    ):
        (snapshot / "big.py").write_text("unexpected write\n")
    assert (path / "big.py").read_text() == "unexpected write\n"
    assert (ops_env.repo / "big.py").read_text().startswith("x = 1")


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"memory_update":null,"proposals":[]}',
        '{"memory_update":null,"memory_update":null}',
        "[]",
        "結果だけ",
        '{"memory_update":{"summary":"missing fields"}}',
    ],
)
def test_parent_final_reply_has_only_one_valid_memory_patch(text: str) -> None:
    with pytest.raises(agent.AgentOutputError):
        agent.parse_completion(text)
    assert agent.parse_completion('{"memory_update":null}') is None


@pytest.mark.parametrize(
    "setting", ["OPS_REPORT_CLEANUP_CMD", "OPS_REPORT_STALE_PR_DAYS"]
)
def test_removed_bulk_collection_settings_are_explicit_errors(
    ops_env: Any, setting: str
) -> None:
    result = ops_env.run("report", **{setting: "old setting"})
    assert result.returncode == 1 and f"{setting} was removed" in result.stderr
    assert not Path(ops_env.env["FAKE_AGENT_USED"]).exists()


def test_publication_before_local_binding_recovers_without_rerunning_agent(
    ops_env: Any,
) -> None:
    assert ops_env.run("report").returncode == 0
    checkpoint = _json(_state_path(ops_env, "audit/20261012.json"))
    path = Path(checkpoint["ledger"])
    document = _json(path)
    assert document["events"][-1]["kind"] == "run.report"
    document["events"].pop()  # crash after GitHub publication, before local bind
    path.write_text(json.dumps(document))
    Path(ops_env.env["FAKE_AGENT_USED"]).unlink()
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 0, result.stderr
    assert not Path(ops_env.env["FAKE_AGENT_USED"]).exists()
    assert len(_reports(ops_env)) == 1
    assert _json(path)["events"][-1]["data"]["report_issue"] == 100
