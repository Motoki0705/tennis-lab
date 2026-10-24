"""Real MCP dispatch with isolated ledger/source state and fake native transcripts."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops/ops-report"))
audit: Any = importlib.import_module("opsreport.audit")
ledger: Any = importlib.import_module("opsreport.ledger")
retrieval: Any = importlib.import_module("opsreport.retrieval")
audit_tools: Any = importlib.import_module("opsreport.audit_tools")
mcp: Any = importlib.import_module("opsreport.mcp_server")
SLOT = "20261012"


def _fake_runtime() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fake_goal_runtime", Path(__file__).with_name("fake_audit_agent.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FakeNative


@pytest.fixture
def tools_env(ops_env: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("CODEX_HOME", ops_env.env["CODEX_HOME"])
    monkeypatch.setenv("PATH", ops_env.env["PATH"])
    state = ops_env.tmp / "tools-state"
    state.mkdir(mode=0o700)
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ops_env.repo, text=True
    ).strip()
    ledger_path = state / f"{SLOT}.ledger.json"
    ledger.Ledger.create(
        ledger_path,
        slot=SLOT,
        base_sha=base,
        goal="目的から必要な問いを作る",
        at=audit.timestamp(),
    )
    input_path = state / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "ops_memory": {
                    "files": {"SUMMARY.md": "PRIVATE_FULL_MEMORY" * 10000},
                    "decisions": [],
                },
                "previous_reports": [],
            }
        )
    )
    runtime = _fake_runtime()(state / "runtime", ops_env.repo)
    server = audit_tools.AuditTools(
        retrieval.Retrieval(ops_env.repo, base, "o/r", SLOT, state / "sources"),
        ledger_path=ledger_path,
        runtime_dir=runtime.runtime,
        input_path=input_path,
        ending_path=state / "ending.json",
        max_threads=3,
    )
    return server, runtime, ops_env


def _call(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    response = mcp.dispatch(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    return json.loads(response["result"]["content"][0]["text"])  # type: ignore[no-any-return]


def _ok(server: Any, name: str, **arguments: Any) -> Any:
    response = _call(server, name, **arguments)
    assert response["ok"], response
    return response["data"]


def _plan(server: Any, name: str, question: str | None = None) -> Any:
    return _ok(
        server,
        "audit_plan_task",
        id=name,
        question=question or f"{name}の利用経路を確認する",
        scope=["big.py"],
        origin={"kind": "goal", "id": SLOT},
    )


def _finish(server: Any, runtime: Any, name: str) -> str:
    runtime.finish_child(name, "確認できた")
    acquired = _call(server, "code_file", path="big.py", start_line=3, line_count=2)
    assert acquired["ok"]
    evidence_id = f"e_{name}"
    _ok(server, "audit_evidence", id=evidence_id, receipt=acquired["receipt"])
    _ok(
        server,
        "audit_finish_task",
        id=name,
        status="completed",
        conclusion="利用経路を確認した",
        evidence=[evidence_id],
        next_questions=[],
        unconfirmed=[],
    )
    return evidence_id


def test_mcp_records_ledger_while_agent_checkout_stays_unchanged(
    tools_env: Any,
) -> None:
    server, runtime, ops = tools_env
    before = subprocess.check_output(["git", "status", "--porcelain"], cwd=ops.repo)
    _plan(server, "question")
    runtime.spawn("question")
    started = _ok(server, "audit_start_task", id="question")
    assert started["agent_id"] == runtime.child_id("question")
    assert started["agent_id"] != "/root/question"
    evidence_id = _finish(server, runtime, "question")
    record = {
        "id": "f",
        "task_id": "question",
        "kind": "design",
        "status": "suspected",
        "conclusion": "利用経路を整理する",
        "evidence": [evidence_id],
        "counterevidence": [],
        "details": {},
    }
    ref = _ok(server, "audit_finding", record=record)["ref"]
    _ok(server, "audit_finding", record={**record, "status": "confirmed"})
    original = _ok(server, "audit_original", ref=ref)
    assert json.loads(original["text"])["status"] == "suspected"
    _ok(
        server,
        "audit_end",
        status="completed",
        reason="重要仮説を確認した",
        checked_scope=["big.py"],
        excluded_scope=[],
        unverified_risks=[],
        further_investigation=[],
    )
    assert server.book().snapshot["run"]["status"] == "running"
    assert not _call(
        server,
        "audit_plan_task",
        id="late",
        question="終了後",
        scope=["big.py"],
        origin={"kind": "goal", "id": SLOT},
    )["ok"]
    runtime.finish()
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert errors == [] and invalid == set()
    ending = json.loads(server.ending_path.read_text())
    assert audit.finish(server.book(), ending, errors, invalid)
    assert (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ops.repo)
        == before
    )
    listing = mcp.dispatch(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    hints = {
        t["name"]: t["annotations"]["readOnlyHint"] for t in listing["result"]["tools"]
    }
    assert hints["code_file"] and hints["audit_status"] and not hints["audit_end"]


def test_existing_thread_pin_limits_running_tasks_and_frees_finished_slot(
    tools_env: Any,
) -> None:
    server, runtime, _ = tools_env
    for name in ("one", "two", "three"):
        _plan(server, name)
        runtime.spawn(name)
        _ok(server, "audit_start_task", id=name)
    _plan(server, "four")
    denied = _call(server, "audit_start_task", id="four")
    assert not denied["ok"] and "OPS_REPORT_MAX_THREADS" in denied["error"]["message"]
    _finish(server, runtime, "one")
    runtime.spawn("four")
    _ok(server, "audit_start_task", id="four")
    assert server.book().snapshot["tasks"]["two"]["status"] == "running"
    assert server.book().snapshot["tasks"]["four"]["status"] == "running"


def test_duplicate_question_and_duplicate_start_are_not_new_work(
    tools_env: Any,
) -> None:
    server, runtime, _ = tools_env
    _plan(server, "one", "利用経路を確認する")
    duplicate = _call(
        server,
        "audit_plan_task",
        id="copy",
        question="  利用経路を確認する  ",
        scope=["big.py"],
        origin={"kind": "goal", "id": SLOT},
    )
    assert (
        not duplicate["ok"]
        and "duplicate question/scope" in duplicate["error"]["message"]
    )
    runtime.spawn("one")
    _ok(server, "audit_start_task", id="one")
    assert not _call(server, "audit_start_task", id="one")["ok"]
    assert len(server.book().snapshot["tasks"]) == 1


@pytest.mark.parametrize("receipt", ["missing", "failed", "ledger"])
def test_missing_failed_and_non_source_receipts_cannot_be_evidence(
    tools_env: Any, receipt: str
) -> None:
    server, _, _ = tools_env
    if receipt == "failed":
        receipt = _call(server, "code_file", path="missing.py")["receipt"]
    elif receipt == "ledger":
        receipt = _call(server, "audit_status")["receipt"]
    response = _call(server, "audit_evidence", id="fake", receipt=receipt)
    assert not response["ok"]
    assert server.book().snapshot["evidence"] == {}


def test_self_reported_success_without_matching_native_execution_is_incomplete(
    tools_env: Any,
) -> None:
    server, runtime, _ = tools_env
    _plan(server, "one")
    runtime.spawn("one")
    _ok(server, "audit_start_task", id="one")
    evidence = _finish(server, runtime, "one")
    finding = {
        "id": "f",
        "task_id": "one",
        "kind": "design",
        "status": "suspected",
        "conclusion": "修正候補",
        "evidence": [evidence],
        "counterevidence": [],
        "details": {},
    }
    _ok(server, "audit_finding", record=finding)
    _ok(server, "audit_finding", record={**finding, "status": "confirmed"})
    runtime.children["one"][2]["payload"]["model"] = "unapproved-model"
    runtime.finish()
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert errors and invalid == {"f"}
    ending = {
        "status": "completed",
        "reason": "自己申告",
        "checked_scope": ["big.py"],
        "excluded_scope": [],
        "unverified_risks": [],
        "further_investigation": [],
    }
    assert not audit.finish(server.book(), ending, errors, invalid)
    assert server.book().snapshot["findings"]["f"]["status"] == "unverified"
    assert server.book().snapshot["handoffs"] == {}


def test_pr_head_changes_preserve_old_receipt_and_require_a_new_acquisition(
    tools_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _, _ = tools_env
    head = "a" * 40
    body = "first version\n" * 200

    def github(argv: list[str], **_kw: Any) -> bytes:
        assert argv[:2] == ["gh", "api"] and "GET" in argv
        return (
            b"HTTP/1.1 200 OK\n\n"
            + json.dumps(
                {
                    "number": 7,
                    "body": body,
                    "head": {"sha": head},
                    "base": {"sha": "c" * 40},
                }
            ).encode()
        )

    monkeypatch.setattr(server.retrieval.command, "run", github)
    original = _call(server, "github_item", kind="pr", number=7, line_count=2)
    assert original["ok"]
    old = _ok(server, "audit_evidence", id="old", receipt=original["receipt"])
    assert old["pr_head"] == "a" * 40
    head = "b" * 40
    body = "updated PR body\n" * 200
    # A continuation fixes the content hash; it must not silently read the new PR.
    changed = _call(server, "github_item", **original["pagination"]["next"])
    assert not changed["ok"] and changed["error"]["code"] == "SOURCE_CHANGED"
    assert not _call(server, "audit_evidence", id="bad", receipt=changed["receipt"])[
        "ok"
    ]
    book = server.book()
    book.bind_report(100, at=audit.timestamp())
    book.end(
        {
            "status": "incomplete",
            "reason": "取得後にPRが更新された",
            "checked_scope": ["取得時のPR #7"],
            "excluded_scope": [],
            "unverified_risks": [
                {
                    "scope": "PR #7の現在head",
                    "reason": "SOURCE_CHANGED",
                    "resume": "新headで取得し直す",
                }
            ],
            "further_investigation": ["新headの再取得"],
        },
        at=audit.timestamp(),
    )
    server.book().resume("同じissueでPR更新後の根拠を取得する", at=audit.timestamp())
    fresh = _call(server, "github_item", kind="pr", number=7, line_count=2)
    _ok(server, "audit_evidence", id="new", receipt=fresh["receipt"])
    book = server.book()
    assert book.snapshot["evidence"]["old"] == old
    assert book.snapshot["evidence"]["new"]["pr_head"] == "b" * 40
    assert book.snapshot["run"]["report_issue"] == 100
    assert book.snapshot["base_sha"] == server.retrieval.base_sha


def test_memory_input_is_bounded_and_does_not_serve_full_snapshot(
    tools_env: Any,
) -> None:
    server, _, _ = tools_env
    response = _call(server, "audit_context", section="ops_memory", max_chars=20)
    assert response["ok"] and len(response["data"]["text"]) <= 20
    assert "PRIVATE_FULL_MEMORY" not in json.dumps(response)


def test_stdio_server_exposes_recording_tools_without_arbitrary_paths_or_commands(
    tools_env: Any,
) -> None:
    server, _, ops = tools_env
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "audit_plan_task",
            "arguments": {
                "id": "stdio",
                "question": "stdio経路の永続化を確認",
                "scope": ["big.py"],
                "origin": {"kind": "goal", "id": SLOT},
            },
        },
    }
    command = [
        sys.executable,
        "-B",
        "-m",
        "opsreport.mcp_server",
        "--repo",
        str(ops.repo),
        "--base-sha",
        server.retrieval.base_sha,
        "--repository",
        "o/r",
        "--round-id",
        SLOT,
        "--journal-dir",
        str(server.retrieval.journal.path.parent),
        "--ledger-path",
        str(server.ledger_path),
        "--runtime-dir",
        str(server.runtime_dir),
        "--input-path",
        str(server.input_path),
        "--ending-path",
        str(server.ending_path),
        "--max-threads",
        "3",
    ]
    result = subprocess.run(
        command,
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / ".agents/ops/ops-report")},
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(json.loads(result.stdout)["result"]["content"][0]["text"])
    assert response["ok"] and "stdio" in server.book().snapshot["tasks"]
    denied = _call(
        server,
        "audit_plan_task",
        id="injected",
        question="x",
        scope=["big.py"],
        origin={"kind": "goal", "id": SLOT},
        command="touch ignored",
    )
    assert not denied["ok"] and denied["error"]["code"] == "INVALID_ARGUMENT"


def test_spawn_interrupted_before_mcp_binding_is_retained_as_unreceived(
    tools_env: Any,
) -> None:
    server, runtime, _ = tools_env
    _plan(server, "unreceived")
    runtime.spawn("unreceived")
    runtime.finish_child("unreceived", "native finished but parent was interrupted")
    runtime.finish()
    book = server.book()
    audit.capture_unreceived(book, runtime.runtime)
    assert book.snapshot["tasks"]["unreceived"]["agent_id"] == runtime.child_id(
        "unreceived"
    )
    audit.interrupt_tasks(book, "親中断で未受領")
    assert book.snapshot["tasks"]["unreceived"]["status"] == "interrupted"
    errors, _ = audit.verify(
        book, [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert errors == []


def test_resume_rejects_live_or_unconfirmed_runner(tools_env: Any) -> None:
    server, runtime, _ = tools_env
    data = json.loads((runtime.runtime / "result.json").read_text())
    data["cleanup_complete"] = False
    (runtime.runtime / "result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cleanup is unconfirmed"):
        audit.ensure_quiescent([str(runtime.runtime)])
    process_identity = importlib.import_module("opsproc").process_identity

    data["cleanup_complete"] = True
    data["runner"] = process_identity(os.getpid())
    (runtime.runtime / "result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="still active"):
        audit.ensure_quiescent([str(runtime.runtime)])


def test_previous_startup_failure_does_not_erase_a_later_verified_success(
    tools_env: Any,
) -> None:
    server, runtime, ops = tools_env
    old = ops.tmp / "startup-failed"
    old.mkdir()
    (old / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result": "failed",
                "turn_end_reason": "startup_failed",
                "contract": {"mode": "read-only", "settings": {"agent": "codex"}},
                "cleanup_complete": True,
                "runtime": {"session_id": None},
            }
        )
    )
    _plan(server, "recovered")
    runtime.spawn("recovered")
    _ok(server, "audit_start_task", id="recovered")
    _finish(server, runtime, "recovered")
    runtime.finish()
    errors, invalid = audit.verify(
        server.book(),
        [str(old), str(runtime.runtime)],
        server.retrieval.journal.path,
        3,
    )
    assert errors == [] and invalid == set()
