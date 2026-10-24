"""Regressions for evidence reachability, MCP-only recovery and interrupted spawns."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

helpers: Any = importlib.import_module(
    "tests.e2e.agents_ops.test_ops_report_audit_tools"
)
cpu: Any = importlib.import_module(
    "tests.e2e.agents_ops.test_ops_report_audit_reproduction"
)
prior_helpers: Any = importlib.import_module(
    "tests.e2e.agents_ops.test_ops_report_handoff"
)
tools_env = helpers.tools_env
scenario = cpu.scenario
connected = cpu.connected
audit, ledger, tools, native = (
    helpers.audit,
    helpers.ledger,
    helpers.audit_tools,
    helpers.audit.native,
)
call, ok, plan, done = helpers._call, helpers._ok, helpers._plan, helpers._finish


def original(server: Any, ref: dict[str, Any]) -> Any:
    arguments = {"ref": ref, "max_chars": 96}
    parts = []
    while True:
        response = call(server, "audit_original", **arguments)
        assert response["ok"], response
        parts.append(response["data"]["text"])
        following = response["pagination"]["next"]
        if following is None:
            return json.loads("".join(parts))
        arguments = {"ref": ref, **following}


def records(server: Any, entity: str, **arguments: Any) -> list[dict[str, Any]]:
    rows = []
    arguments = {"entity": entity, "per_page": 1, **arguments}
    while True:
        response = call(server, "audit_records", **arguments)
        assert response["ok"], response
        rows.extend(response["data"])
        following = response["pagination"]["next"]
        if following is None:
            assert response["pagination"]["total"] == len(rows)
            return rows
        arguments = following


def bug(server: Any, task: str, main: str, usage: str) -> dict[str, Any]:
    row = {
        "id": "usage_only",
        "task_id": task,
        "kind": "bug",
        "status": "suspected",
        "conclusion": "A reachable bug requires both source and usage evidence",
        "evidence": [main],
        "counterevidence": [],
        "details": {
            "target_version": server.retrieval.base_sha,
            "conditions": ["CLI entry"],
            "reachability": "config selects this entry",
            "impact": "wrong result",
            "remediation": "correct the branch",
            "verification": "check the entry",
            "usage_paths": [
                {
                    "kind": "config-cli",
                    "outcome": "used",
                    "evidence": [usage],
                    "reason": "checked CLI",
                }
            ],
        },
    }
    ok(server, "audit_finding", record=row)
    row["status"] = "confirmed"
    ok(server, "audit_finding", record=row)
    return row


def test_missing_receipt_used_only_by_usage_path_invalidates_its_finding(
    tools_env: Any,
) -> None:
    server, runtime, _ = tools_env
    plan(server, "question")
    runtime.spawn("question")
    ok(server, "audit_start_task", id="question")
    main = done(server, runtime, "question")
    acquired = call(server, "code_file", path="big.py", start_line=7, line_count=1)
    assert acquired["ok"]
    ok(server, "audit_evidence", id="usage", receipt=acquired["receipt"])
    bug(server, "question", main, "usage")
    runtime.finish()
    journal = server.retrieval.journal.path
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    journal.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in rows
            if not (
                row.get("event") == "received" and row["receipt"] == acquired["receipt"]
            )
        )
    )
    errors, invalid = audit.verify(server.book(), [str(runtime.runtime)], journal, 3)
    assert errors and invalid == {"usage_only"}
    assert not audit.finish(server.book(), None, errors, invalid)
    assert server.book().snapshot["findings"]["usage_only"]["status"] == "unverified"
    assert not server.book().snapshot["handoffs"]


def test_failed_cpu_evidence_used_only_by_usage_path_cannot_confirm(
    connected: Any,
) -> None:
    s, server, runtime, _ = connected
    s.runner.exit_code = 1
    acquired = call(server, "code_file", path="tests/test_cpu.py", line_count=2)
    assert acquired["ok"]
    ok(server, "audit_evidence", id="main", receipt=acquired["receipt"])
    response = call(
        server, "cpu_tests", finding_id="finding", nodeids=[cpu.repro_tests.NODE]
    )
    assert response["ok"] and response["data"]["disposition"] == "needs-reproduction"
    bug(server, "task", "main", response["data"]["evidence_id"])
    cpu.finish(server, runtime, "main")
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert any("needs-reproduction" in e for e in errors)
    assert invalid == {"finding", "usage_only"}
    assert not audit.finish(server.book(), None, errors, invalid)
    assert server.book().snapshot["findings"]["usage_only"]["status"] == "unverified"
    assert not server.book().snapshot["handoffs"]


def test_completed_tasks_and_confirmed_refuted_findings_have_paged_versioned_refs(
    tools_env: Any,
) -> None:
    server, runtime, _ = tools_env
    for name in ["first", "second"]:
        plan(server, name)
        runtime.spawn(name)
        ok(server, "audit_start_task", id=name)
        evidence = done(server, runtime, name)
        row = {
            "id": name,
            "task_id": name,
            "kind": "design",
            "status": "suspected",
            "conclusion": "long finding " * 60,
            "evidence": [evidence],
            "counterevidence": [],
            "details": {},
        }
        ok(server, "audit_finding", record=row)
        ok(
            server,
            "audit_finding",
            record={
                **row,
                "status": "confirmed" if name == "first" else "refuted",
                "counterevidence": [] if name == "first" else [evidence],
            },
        )
    status = ok(server, "audit_status")
    assert not status["unfinished_tasks"] and not status["unverified_hypotheses"]
    task_rows = records(server, "task")
    findings = records(server, "finding")
    assert {r["id"] for r in task_rows} == {"first", "second"}
    assert {original(server, r["ref"])["status"] for r in findings} == {
        "confirmed",
        "refuted",
    }
    assert all(original(server, r["ref"])["status"] == "completed" for r in task_rows)
    old_ref = next(r["ref"] for r in findings if r["id"] == "first")
    row = original(server, old_ref)
    ok(
        server,
        "audit_finding",
        record={**row, "status": "refuted", "counterevidence": row["evidence"]},
    )
    assert original(server, old_ref)["status"] == "confirmed"
    assert not call(server, "audit_records", entity="task", page=99, per_page=1)["ok"]


@pytest.fixture
def prior(tools_env: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    server, _, _ = tools_env
    slot = "20261009"
    path = server.ledger_path.with_name(f"{slot}.ledger.json")
    book = prior_helpers._book(path, slot=slot, base="b" * 40)
    prior_helpers._finding(book, "previous_problem")
    prior_helpers._end(book)
    book.bind_report(909, at=prior_helpers.AT)
    _, body, _ = prior_helpers._render(book)

    def gh(args: list[str], **kwargs: Any) -> dict[str, str]:
        assert args == ["issue", "view", "909", "--json", "body"]
        return {"body": body}

    monkeypatch.setattr(tools.shell, "gh_json", gh)
    pinned = json.loads(server.input_path.read_text())
    pinned["previous_reports"] = [
        {
            "slot": slot,
            "number": 909,
            "base_sha": "b" * 40,
            "status": "complete",
            "url": "https://github.com/o/r/issues/909",
        }
    ]
    server.input_path.write_text(json.dumps(pinned))
    return server, book, slot


def test_previous_handoffs_resolve_originals_and_evidence_in_the_pinned_round(
    prior: Any,
) -> None:
    server, _, slot = prior
    summary = ok(server, "audit_previous_handoffs", slot=slot, per_page=1)
    assert summary["source_slot"] == slot and summary["items"]
    for key in ["finding_ref", "handoff_ref"]:
        assert original(server, summary["items"][0][key])["id"]
    assert original(server, summary["run_ref"])["report_issue"] == 909
    evidence = records(server, "evidence", slot=slot)
    assert original(server, evidence[0]["ref"])["target_version"] == "b" * 40


@pytest.mark.parametrize(
    "problem",
    [
        "arbitrary_path",
        "unlisted_round",
        "wrong_round",
        "corrupt",
        "symlink",
        "unsafe_slot",
        "duplicate_pin",
        "incomplete_pin",
        "report_mismatch",
        "base_mismatch",
        "ref_hash",
    ],
)
def test_originals_never_escape_fixed_previous_reports(
    prior: Any, problem: str
) -> None:
    server, book, slot = prior
    ref = book.reference("finding", "previous_problem")
    if problem == "arbitrary_path":
        ref["path"] = str(book.path.parent / ".." / book.path.name)
    elif problem == "unlisted_round":
        pinned = json.loads(server.input_path.read_text())
        pinned["previous_reports"] = []
        server.input_path.write_text(json.dumps(pinned))
    elif problem == "wrong_round":
        doc = json.loads(book.path.read_text())
        doc["slot"] = "20261006"
        book.path.write_text(json.dumps(doc))
    elif problem in {
        "duplicate_pin",
        "incomplete_pin",
        "report_mismatch",
        "base_mismatch",
    }:
        pinned = json.loads(server.input_path.read_text())
        row = pinned["previous_reports"][0]
        if problem == "duplicate_pin":
            pinned["previous_reports"].append(dict(row))
        elif problem == "incomplete_pin":
            row["status"] = "incomplete"
        elif problem == "report_mismatch":
            row["number"] = 910
        else:
            row["base_sha"] = "c" * 40
        server.input_path.write_text(json.dumps(pinned))
    elif problem == "ref_hash":
        ref["sha256"] = "f" * 64
    elif problem == "corrupt":
        book.path.write_text("{broken")
    elif problem == "symlink":
        moved = book.path.with_suffix(".saved")
        book.path.rename(moved)
        book.path.symlink_to(moved)
    else:
        assert not call(server, "audit_previous_handoffs", slot="../escape")["ok"]
        return
    assert not call(server, "audit_original", ref=ref)["ok"]


def interrupted(runtime: Any) -> None:
    runtime.rows = [
        row
        for row in runtime.rows
        if not (
            row["type"] == "response_item"
            and row["payload"].get("type") == "function_call_output"
            and row["payload"]["call_id"].startswith("spawn-retry-")
        )
    ]
    runtime.result.update(
        result="failed", turn_end_reason="stopped", exit_code=143, cli_exit_code=-15
    )
    runtime.save()


def test_unpaired_interrupted_spawn_recovers_without_losing_a_verified_child(
    tools_env: Any,
) -> None:
    server, previous, ops = tools_env
    for name in ["verified", "retry"]:
        plan(server, name)
        previous.spawn(name)
        if name == "verified":
            ok(server, "audit_start_task", id=name)
            done(server, previous, name)
    interrupted(previous)
    with pytest.raises(native.NativeEvidenceError, match="unpaired"):
        native.inspect_run(previous.runtime)
    errors, invalid = audit.verify(
        server.book(), [str(previous.runtime)], server.retrieval.journal.path, 3
    )
    assert errors
    assert not audit.finish(server.book(), None, errors, invalid)
    original_transcript = (
        previous.directory / f"rollout-{previous.parent}.jsonl"
    ).read_bytes()
    audit.ensure_quiescent([str(previous.runtime)])
    server.book().resume("retry the interrupted question", at=audit.timestamp())
    current = helpers._fake_runtime()(previous.runtime.parent / "next", ops.repo)
    recovered = tools.AuditTools(
        server.retrieval,
        ledger_path=server.ledger_path,
        runtime_dir=current.runtime,
        input_path=server.input_path,
        ending_path=server.ending_path,
        max_threads=3,
    )
    current.spawn("retry")
    ok(recovered, "audit_start_task", id="retry")
    done(recovered, current, "retry")
    current.finish()
    errors, invalid = audit.verify(
        recovered.book(),
        [str(previous.runtime), str(current.runtime)],
        server.retrieval.journal.path,
        3,
    )
    assert errors == [] and invalid == set()
    rows = records(recovered, "task")
    assert original(recovered, next(r["ref"] for r in rows if r["id"] == "verified"))[
        "agent_id"
    ] == previous.child_id("verified")
    assert (
        previous.directory / f"rollout-{previous.parent}.jsonl"
    ).read_bytes() == original_transcript
    assert recovered.book().snapshot["run"]["endings"][0]["status"] == "incomplete"


@pytest.mark.parametrize(
    "problem",
    [
        "running",
        "cleanup",
        "live",
        "exit_unknown",
        "bad_model",
        "corrupt",
        "missing_process",
        "wrong_reason",
        "completed_old",
        "bad_exit_type",
    ],
)
def test_only_terminated_clean_interrupted_attempts_can_be_kept_as_history(
    tools_env: Any, problem: str
) -> None:
    server, previous, ops = tools_env
    plan(server, "retry")
    previous.spawn("retry")
    interrupted(previous)
    if problem == "running":
        previous.result["result"] = "running"
    elif problem == "cleanup":
        previous.result["cleanup_complete"] = False
    elif problem == "live":
        previous.result["process"] = importlib.import_module(
            "opsproc"
        ).process_identity(os.getpid())
    elif problem == "exit_unknown":
        previous.result["cli_exit_code"] = None
    elif problem == "bad_model":
        previous.children["retry"][2]["payload"]["model"] = "other-model"
    elif problem == "missing_process":
        previous.result.pop("process")
    elif problem == "wrong_reason":
        previous.result["turn_end_reason"] = "metadata_error"
    elif problem == "completed_old":
        previous.finish()
    elif problem == "bad_exit_type":
        previous.result["exit_code"] = 143.0
    previous.save()
    current = helpers._fake_runtime()(previous.runtime.parent / "next", ops.repo)
    next_server = tools.AuditTools(
        server.retrieval,
        ledger_path=server.ledger_path,
        runtime_dir=current.runtime,
        input_path=server.input_path,
        ending_path=server.ending_path,
        max_threads=3,
    )
    current.spawn("retry")
    ok(next_server, "audit_start_task", id="retry")
    done(next_server, current, "retry")
    current.finish()
    if problem == "corrupt":
        (previous.directory / f"rollout-{previous.parent}.jsonl").write_text(
            "{broken\n"
        )
    errors, _ = audit.verify(
        next_server.book(),
        [str(previous.runtime), str(current.runtime)],
        server.retrieval.journal.path,
        3,
    )
    assert errors


@pytest.mark.parametrize("interruption", ["child_failure", "unpaired"])
def test_mcp_only_resume_reuses_completed_records_after_failure_or_interruption(
    ops_env: Any,
    interruption: str,
) -> None:
    injected = (
        {"FAKE_AUDIT_FAIL": "independent"}
        if interruption == "child_failure"
        else {"FAKE_AUDIT_UNPAIRED": "1"}
    )
    failed = ops_env.run("report", **injected, FAKE_AUDIT_RECORD_PAGE_SIZE="2")
    assert failed.returncode == 1, failed.stderr
    report = next(r for r in ops_env.state["issues"] if "ops-report" in r["labels"])
    assert "status=incomplete" in report["body"]
    first = max(ops_env.run_dirs(), key=lambda p: p.stat().st_mtime_ns)
    connection = json.loads((first / "mcp.json").read_text())["ops_report"]["args"]
    source_dir = Path(connection[connection.index("--journal-dir") + 1])
    journal = source_dir / "retrieval.jsonl"
    before = journal.read_text().splitlines()
    first_received = [
        json.loads(line)["response"]
        for line in before
        if json.loads(line)["event"] == "received"
    ]
    first_originals = [
        json.loads(r["data"]["text"])
        for r in first_received
        if r["ok"]
        and isinstance(r["data"], dict)
        and "text" in r["data"]
        and r["data"]["text"].startswith('{"id":')
    ]
    verified = next(
        row
        for row in first_originals
        if row["id"] == "initial" and row["status"] == "completed"
    )
    resumed = ops_env.run(
        "report",
        "--if-due",
        FAKE_AUDIT_RECORD_PAGE_SIZE="2",
        FAKE_AUDIT_DISCOVERY="dependency",
    )
    assert resumed.returncode == 0, resumed.stderr
    published = [r for r in ops_env.state["issues"] if "ops-report" in r["labels"]]
    assert len(published) == 1 and published[0]["number"] == report["number"]
    assert "status=complete" in published[0]["body"]
    after = [
        json.loads(line) for line in journal.read_text().splitlines()[len(before) :]
    ]
    received = [r["response"] for r in after if r["event"] == "received"]
    acquired = [
        json.loads(r["data"]["text"])
        for r in received
        if r["ok"]
        and isinstance(r["data"], dict)
        and "text" in r["data"]
        and r["data"]["text"].startswith('{"id":')
    ]
    assert any(row == verified for row in acquired)
    assert {row["status"] for row in acquired if row["id"] == "initial_finding"} == (
        {"refuted"} if interruption == "child_failure" else {"unverified", "refuted"}
    )
    assert any(row["id"] == "f0" and row["status"] == "confirmed" for row in acquired)
    assert any(
        r["event"] == "requested"
        and r["tool"] == "audit_records"
        and r["arguments"].get("page", 1) > 1
        for r in after
    )
    latest = max(ops_env.run_dirs(), key=lambda p: p.stat().st_mtime_ns)
    assert [
        a.task_name for a in native.inspect_run(latest / "agent-runtime").agents
    ] == [
        "/root/independent"
        if interruption == "child_failure"
        else "/root/interrupted_launch"
    ]
    if interruption == "unpaired":
        historical = native.inspect_run(
            first / "agent-runtime", allow_interrupted_spawn=True
        )
        assert [s.task_name for s in historical.unpaired_spawns] == [
            "interrupted_launch"
        ]


@pytest.mark.parametrize("state", ["completed", "stopped"])
def test_current_attempt_never_accepts_an_unpaired_spawn_as_completed(
    tools_env: Any, state: str
) -> None:
    server, runtime, _ = tools_env
    plan(server, "retry")
    runtime.spawn("retry")
    ok(server, "audit_start_task", id="retry")
    done(server, runtime, "retry")
    interrupted(runtime)
    if state == "completed":
        runtime.finish()
    errors, invalid = audit.verify(
        server.book(), [str(runtime.runtime)], server.retrieval.journal.path, 3
    )
    assert any("unpaired" in e for e in errors)
    assert any("completed task has no completed native" in e for e in errors)
    assert not invalid
