"""Goal-run MCP composes the shared CPU executor with ledger/native verification."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops/ops-report"))
repro_tests: Any = importlib.import_module(
    "tests.e2e.agents_ops.test_ops_report_reproduction"
)
scenario = repro_tests.scenario
mcp: Any = importlib.import_module("opsreport.mcp_server")
audit: Any = importlib.import_module("opsreport.audit")
audit_tools: Any = importlib.import_module("opsreport.audit_tools")
cli: Any = importlib.import_module("opsreport.cli")
shared: Any = importlib.import_module("lib.review_tools.mcp_server")


@pytest.fixture
def connected(scenario: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    s = scenario
    monkeypatch.setenv("CODEX_HOME", str(s.path.parent / "codex-home"))
    allowlist = s.path.parent / "cpu-tests.json"
    allowlist.write_text(
        json.dumps({"base_sha": s.base, "nodeids": [repro_tests.NODE]})
    )
    args = argparse.Namespace(
        repo=s.repo,
        base_sha=s.base,
        repository="o/r",
        round_id=repro_tests.SLOT,
        journal_dir=s.path.parent / "sources",
        ledger_path=s.book,
        ops_memory_sha=None,
        cpu_tests_file=allowlist,
        isolation_proof=None,
    )
    retrieval = mcp.source(args, runner=s.runner)
    retrieval.reproduction.execution_lock = s.lock
    runtime = importlib.import_module(
        "tests.e2e.agents_ops.test_ops_report_audit_tools"
    )._fake_runtime()(s.path.parent / "runtime", s.repo)
    source_input = s.path.parent / "input.json"
    source_input.write_text('{"ops_memory":{},"previous_reports":[]}')
    server = audit_tools.AuditTools(
        retrieval,
        ledger_path=s.book,
        runtime_dir=runtime.runtime,
        input_path=source_input,
        ending_path=s.path.parent / "ending.json",
        max_threads=3,
    )
    runtime.spawn("task")
    assert server.call("audit_start_task", {"id": "task"})["ok"]
    return s, server, runtime, args


def finish(server: Any, runtime: Any, evidence: str) -> None:
    runtime.finish_child("task", "test result received")
    result = server.call(
        "audit_finish_task",
        {
            "id": "task",
            "status": "completed",
            "conclusion": "received",
            "evidence": [evidence],
            "next_questions": [],
            "unconfirmed": [],
        },
    )
    assert result["ok"], result
    runtime.finish()


def test_goal_mcp_uses_shared_transport_and_executor_and_verifies_cpu_records(
    connected: Any,
) -> None:
    s, server, runtime, _ = connected
    assert mcp.dispatch is shared.dispatch and mcp.serve is shared.serve
    assert isinstance(
        server.retrieval.reproduction,
        importlib.import_module("lib.review_tools.reproduction").Reproduction,
    )
    response = server.call(
        "cpu_tests", {"finding_id": "finding", "nodeids": [repro_tests.NODE]}
    )
    assert response["ok"], response
    assert response["data"]["status"] == "passed"
    key = response["data"]["evidence_id"]
    finish(server, runtime, key)
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert errors == [] and invalid == set()
    assert (
        s.runner.jobs
        and server.book().snapshot["findings"]["finding"]["status"] == "suspected"
    )
    listing = mcp.dispatch(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    hints = {
        t["name"]: t["annotations"]["readOnlyHint"] for t in listing["result"]["tools"]
    }
    assert hints["cpu_test_status"] and not hints["cpu_tests"]


@pytest.mark.parametrize("damage", ["manifest", "log", "receipt"])
def test_missing_cpu_evidence_stays_incomplete_and_recovers_from_original_records(
    connected: Any, damage: str
) -> None:
    _, server, runtime, _ = connected
    response = server.call(
        "cpu_tests", {"finding_id": "finding", "nodeids": [repro_tests.NODE]}
    )
    assert response["ok"], response
    key = response["data"]["evidence_id"]
    book = server.book()
    finding = book.snapshot["findings"]["finding"]
    book.record_finding({**finding, "status": "confirmed"}, at=audit.timestamp())
    finish(server, runtime, key)
    directory = server.retrieval.reproduction.state_dir / response["data"]["run_id"]
    target = (
        server.retrieval.journal.path
        if damage == "receipt"
        else directory / ("result.json" if damage == "manifest" else "pytest.log")
    )
    original = target.read_bytes()
    if damage == "receipt":
        rows = original.splitlines()
        target.write_bytes(
            b"\n".join(
                rows[:1]
                + [r for r in rows[1:] if response["receipt"].encode() not in r]
            )
            + b"\n"
        )
    else:
        target.unlink()
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert errors and invalid == {"finding"}
    ending = {
        "status": "completed",
        "reason": "test",
        "checked_scope": ["tests"],
        "excluded_scope": [],
        "unverified_risks": [],
        "further_investigation": [],
    }
    assert not audit.finish(server.book(), ending, errors, invalid)
    assert server.book().snapshot["findings"]["finding"]["status"] == "unverified"
    target.write_bytes(original)
    server.book().resume("restore original evidence", at=audit.timestamp())
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert errors == [] and invalid == set()


def test_failed_cpu_result_is_never_a_confirmed_finding(connected: Any) -> None:
    s, server, runtime, _ = connected
    s.runner.exit_code = 1
    response = server.call(
        "cpu_tests", {"finding_id": "finding", "nodeids": [repro_tests.NODE]}
    )
    assert response["ok"] and response["data"]["disposition"] == "needs-reproduction"
    book = server.book()
    book.record_finding(
        {**book.snapshot["findings"]["finding"], "status": "confirmed"},
        at=audit.timestamp(),
    )
    finish(server, runtime, response["data"]["evidence_id"])
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert any("needs-reproduction" in e for e in errors) and invalid == {"finding"}


def test_missing_proof_disables_execution_and_wrong_revision_is_explicit(
    connected: Any,
) -> None:
    _, _, _, args = connected
    source = mcp.source(args)
    assert "cpu_tests" not in source.tools
    status = source.call("cpu_test_status", {})
    assert (
        status["ok"] and not status["data"]["enabled"] and status["data"]["unverified"]
    )
    rejected = source.call(
        "cpu_tests", {"finding_id": "finding", "nodeids": [repro_tests.NODE]}
    )
    assert not rejected["ok"] and rejected["error"]["code"] == "ISOLATION_UNAVAILABLE"
    args.cpu_tests_file.write_text(
        json.dumps({"base_sha": "f" * 40, "nodeids": [repro_tests.NODE]})
    )
    with pytest.raises(ValueError, match="fixed base SHA"):
        mcp.source(args)


def test_cpu_config_is_pinned_outside_prompt_and_not_reread_from_mutable_allowlist(
    connected: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    s, _, _, args = connected
    run = s.path.parent / "launch"
    run.mkdir()
    monkeypatch.setenv("OPS_REPORT_CPU_TESTS_FILE", str(args.cpu_tests_file))
    monkeypatch.setenv("OPS_REPORT_ISOLATION_PROOF", str(s.path.parent / "proof.json"))
    result = cli._cpu_arguments(run, s.base)
    assert result == [
        "--cpu-tests-file",
        str(run / "cpu-tests.json"),
        "--isolation-proof",
        str(s.path.parent / "proof.json"),
    ]
    original = (run / "cpu-tests.json").read_bytes()
    args.cpu_tests_file.write_text("changed after launch")
    assert (run / "cpu-tests.json").read_bytes() == original
