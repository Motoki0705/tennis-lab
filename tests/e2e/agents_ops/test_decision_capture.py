"""Observe real parent/review entry points with fake GitHub and disposable git."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import (
    manual,
    planning_answer,
    resume_answer,
    run,
    state_dir,
)
from tests.e2e.agents_ops.test_manual_worker_runs import (
    reserved as reserved,
)
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    ReviewEnv,
    evidence_number,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.support.agents_ops.decision_logs import graph

C = importlib.import_module("decision_logs.capture")


def captured(directory: Path) -> list[dict[str, Any]]:
    value = C.status(directory, "o/r", 1)
    assert value["status"] == "captured", value
    assert all(r["view_status"] == "ready" for r in value["records"]), value
    return [r["record"] for r in value["records"]]


def sourced(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r["kind"] == "result"
        and any(
            f"capture_source={name};" in e.get("reason", "")
            for e in r["data"]["evidence"]
        )
    ]


def test_delegate_private_identity_is_guarded_before_worker_launch(
    reserved: ParentEnv,
) -> None:
    p = reserved
    worker_tests = importlib.import_module(
        "tests.e2e.agents_ops.test_manual_worker_runs"
    )
    task = worker_tests.task(p)
    private_id = task["delegation_id"]
    assert not task["worker_runs"]
    view = json.loads(p.cli("decision-log-status").stdout)
    assert view["status"] == "captured"
    assert private_id not in json.dumps(view)
    journal = C.Journal(state_dir(p), "o/r", 1)
    entries = [e for e in journal.read()["entries"] if e["event"] == "capture"]
    delegated = [
        e
        for e in entries
        if e["record"]["kind"] == "result"
        and e["record"]["data"]["summary"].startswith("parent.delegate:")
    ]
    assert delegated and all(
        private_id in e["private"]["identifiers"] for e in delegated
    )
    assert all("assignment" not in e["record"]["data"]["summary"] for e in delegated)
    for entry in entries:
        journal.prepare(
            entry["record"]["id"], key="public:" + entry["record"]["id"], at=C.now()
        )
    payloads = journal.public_payloads()
    assert payloads and private_id not in json.dumps(payloads)


def test_worker_planning_judgement_continuation_and_explicit_verification(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p, "--planning")
    answer = json.loads(planning_answer(p).stdout)
    rows = captured(state_dir(p))
    runtime_result = next(
        r
        for r in reversed(rows)
        if r["kind"] == "result" and "turn_end_reason=completed" in r["data"]["summary"]
    )
    planning_result = sourced(rows, "parent.decide-worker-plan")[-1]
    assert (
        runtime_result["data"]["auxiliary"]["selected_model"]["value"]["specified_by"]
        == "settings"
    )
    assert (
        graph.states(rows)[runtime_result["id"]]["parent_confirmation"] == "unrecorded"
    )
    spec = p.root / "decision.json"
    spec.write_text(
        json.dumps(
            {
                "inputs": [runtime_result["id"], planning_result["id"]],
                "evaluation": "continue",
                "reason_ref": {
                    "kind": "worker_plan",
                    "child": 10,
                    "decision_id": answer["decision_id"],
                },
                "next_actions": [
                    {
                        "operation": "continue_ai",
                        "summary": "Implement the accepted plan",
                        "actor": "worker",
                        "target": "child 10",
                        "instruction": "Use the saved answer",
                        "conditions": "Same session and model",
                        "expected_result": "A tested PR",
                    }
                ],
                "reuse_scope": ["planning evidence"],
                "recheck_scope": ["implementation tests"],
            }
        )
    )
    decision = json.loads(p.cli("decide", "--spec", str(spec)).stdout)
    assert json.loads(p.cli("decide", "--spec", str(spec)).stdout) == decision
    resume_answer(p, "--decision-id", decision["decision_id"])
    rows = captured(state_dir(p))
    judgement = next(r for r in rows if r["id"] == decision["decision_id"])
    assert judgement["data"]["inputs"] == [runtime_result["id"], planning_result["id"]]
    assert "See existing worker_plan" in judgement["data"]["reason"]
    actions = [
        r
        for r in rows
        if r["kind"] == "action" and r["data"]["decision_id"] == decision["decision_id"]
    ]
    assert actions[0]["data"]["operation"] == "continue_ai"
    assert actions[0]["data"]["plan_id"] == decision["plan_ids"][0]
    following = [
        r
        for r in rows
        if r["kind"] == "result"
        and r["data"]["action_id"] in {a["id"] for a in actions}
    ]
    assert following[0]["data"]["previous_result_id"] == runtime_result["id"]
    assert following[0]["data"]["changes"]["instruction"] == "changed"
    assert following[0]["data"]["changes"]["requirements"] == "unchanged"
    assert all(
        s["parent_confirmation"] == "unrecorded"
        for s in graph.states(
            following + [r for r in rows if r not in following]
        ).values()
    )
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/worker")]
    p.save(state)
    manual(
        p,
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("PRと回帰テストを確認した。"),
    )
    rows = captured(state_dir(p))
    checked = sourced(rows, "parent.record-result")[-1]
    assert (
        checked["data"]["observer"] == "parent_verification"
        and checked["data"]["verifies"]
    )
    assert all(
        graph.states(rows)[key]["parent_confirmation"] == "recorded"
        for key in checked["data"]["verifies"]
    )


@pytest.mark.parametrize(
    "failure", ["input", "runner_rejection", "runtime_failure", "final_missing"]
)
def test_failures_never_become_success_or_parent_acceptance(
    reserved: ParentEnv, failure: str
) -> None:
    p = reserved
    if failure == "input":
        response = manual(
            p, "run-worker", "--child", "999", "--prompt-file", p.note(), ok=False
        )
        expected = 2
    elif failure == "final_missing":
        response = run(p, MANUAL_PLANNING_FINAL="")
        expected = 0
    else:
        response = run(
            p,
            ok=False,
            **(
                {"MANUAL_PREFLIGHT_FAILURE": "1"}
                if failure == "runner_rejection"
                else {"MANUAL_RUNNER_EXIT": "17"}
            ),
        )
        expected = 2 if failure == "runner_rejection" else 17
    assert response.returncode == expected
    rows = captured(state_dir(p))
    results = sourced(rows, "parent.run-worker")
    assert results[-1]["data"]["status"] in {"failure", "rejected", "partial"}
    if failure == "final_missing":
        assert not any(r["data"]["status"] == "success" for r in results)
        assert any(
            r["data"]["final_report"]
            == {"status": "unavailable", "reason": "missing_final"}
            for r in results
        )
    assert not any(
        r["kind"] == "result" and r["data"]["observer"] == "parent_verification"
        for r in rows
    )
    actions = [
        r
        for r in rows
        if r["kind"] == "action" and r["data"]["operation"] == "start_ai"
    ]
    assert actions[-1]["data"]["state"] == (
        "ended"
        if failure in {"runtime_failure", "final_missing"}
        else "rejected_before_start"
    )
    assert all(
        a["data"]["decision_id"] is None and a["data"]["decision_missing_reason"]
        for a in actions
    )


def test_note_reference_stop_resume_handoff_and_release_are_captured(
    reserved: ParentEnv,
) -> None:
    p = reserved
    note = json.loads(
        p.cli("note", "--body-file", p.note("入力が揃うまで保留する。")).stdout
    )
    p.cli("stop", "--child", "10", "--spec", p.detail())
    rows = captured(state_dir(p))
    spec = p.root / "decision.json"
    spec.write_text(
        json.dumps(
            {
                "inputs": [sourced(rows, "parent.stop")[-1]["id"]],
                "evaluation": "wait",
                "reason_ref": {"kind": "note", "body_sha256": note["body_sha256"]},
                "resume_conditions": ["inputs become available"],
            }
        )
    )
    value = json.loads(p.cli("decide", "--spec", str(spec)).stdout)
    rows = captured(state_dir(p))
    decision = next(r for r in rows if r["id"] == value["decision_id"])
    assert decision["data"]["source"]["evidence"][0]["url"].startswith(
        f"https://github.com/o/r/issues/{evidence_number(p.state())}#issuecomment-"
    )
    p.cli("resume", "--child", "10", "--spec", p.detail())
    p.cli("handoff", "--note-file", p.note())
    p.env["CLAUDE_CODE_SESSION_ID"] = "next-parent-session"
    p.cli("handoff", "--accept", "--note-file", p.note())
    p.cli("release", "--note-file", p.note())
    rows = captured(state_dir(p))
    for entrypoint in (
        "parent.note",
        "parent.stop",
        "parent.resume",
        "parent.handoff",
        "parent.release",
    ):
        assert sourced(rows, entrypoint)


def test_review_execution_publication_reuse_and_explicit_receipt(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    before = captured(state_dir(r.p))
    requested = sourced(before, "parent.review-request")[-1]
    decision = json.loads(
        r.p.cli(
            "decide",
            "--spec",
            r.p.spec(
                {
                    "inputs": [requested["id"]],
                    "evaluation": "investigate",
                    "reason": "Review the fixed targets before accepting the result",
                    "next_actions": [
                        {
                            "operation": "review",
                            "summary": "Review fixed PRs",
                            "actor": "external",
                            "target": "PR 100",
                            "instruction": "Use the fixed request",
                            "conditions": "Pinned targets",
                            "expected_result": "A review result",
                        }
                    ],
                }
            ),
        ).stdout
    )
    r.p.cli(
        "review-request",
        "--run-id",
        r.request["review_run_id"],
        "--instruction-file",
        r.p.note("固定した要求でreviewを実行する。"),
        "--decision-id",
        decision["decision_id"],
    )
    r.run()
    initial = captured(state_dir(r.p))
    linked = {
        v["id"]
        for v in initial
        if v["kind"] == "action" and v["data"]["decision_id"] == decision["decision_id"]
    }
    assert linked and all(
        v["data"]["action_id"] in linked for v in sourced(initial, "review.run")
    )
    raw = [
        v
        for v in sourced(initial, "review.run")
        if "review.result.json:" in v["data"]["summary"]
    ][-1]
    assert raw["data"]["status"] == "success"
    assert any(
        "status=published" in v["data"]["summary"]
        for v in sourced(initial, "review.run")
    )
    r.run()
    rows = captured(state_dir(r.p))
    roots = [
        v["data"]["operation"]
        for v in rows
        if v["kind"] == "action" and v["data"]["previous_action_id"] is None
    ]
    assert roots.count("fetch") == 1
    assert len(r.calls()) == 1
    assert all(
        s["parent_confirmation"] == "unrecorded" for s in graph.states(rows).values()
    )
    r.p.cli(
        "review-state",
        "--run-id",
        r.request["review_run_id"],
        "--state",
        "result_received",
        "--evidence-file",
        r.p.note("レビューを受領した。"),
    )
    rows = captured(state_dir(r.p))
    received = sourced(rows, "parent.review-state")[-1]
    assert received["data"]["observer"] == "parent_verification"
    assert received["data"]["verifies"]
    assert sourced(rows, "parent.review-request")


@pytest.mark.parametrize("exit_code", [0, 17])
@pytest.mark.parametrize("failure_mode", ["journal", "outbox_lock"])
def test_journal_fault_never_changes_worker_exit_or_launches_a_retry(
    reserved: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    failure_mode: str,
) -> None:
    worker_tests = importlib.import_module(
        "tests.e2e.agents_ops.test_manual_worker_runs"
    )
    injection = """
from decision_logs.capture import Journal
def broken_capture(*args, **kwargs):
    raise OSError('injected journal failure')
Journal.capture = broken_capture
"""
    if failure_mode == "outbox_lock":
        injection = """
from contextlib import contextmanager
from decision_logs import capture as capture_module
saved_lock = capture_module.locked
@contextmanager
def broken_outbox_lock(path):
    if path.parent.name == 'captures':
        raise OSError('injected outbox lock failure')
    with saved_lock(path):
        yield
capture_module.locked = broken_outbox_lock
"""
    monkeypatch.setattr(
        worker_tests,
        "ENTRY",
        worker_tests.ENTRY.replace(
            "from parent.cli import main", injection + "\nfrom parent.cli import main"
        ),
    )
    p = reserved
    response = run(p, ok=exit_code == 0, MANUAL_RUNNER_EXIT=str(exit_code))
    assert response.returncode == exit_code
    assert "capture incomplete" in response.stderr
    before = (p.root / "runner-calls.jsonl").read_bytes()
    status_response = p.cli("decision-log-status", ok=False)
    assert json.loads(status_response.stdout)["status"] == "incomplete"
    recovered = json.loads(
        p.cli("retry-decision-capture", ok=failure_mode == "journal").stdout
    )
    assert recovered["status"] == (
        "captured" if failure_mode == "journal" else "incomplete"
    )
    assert (p.root / "runner-calls.jsonl").read_bytes() == before
    if failure_mode == "journal":
        assert captured(state_dir(p))
    assert worker_tests.task(p)["worker_runs"] and worker_tests.task(p)["worker_ended"]


def test_worker_decision_lookup_io_failure_is_not_invalid_input(
    reserved: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = reserved
    worker_tests = importlib.import_module(
        "tests.e2e.agents_ops.test_manual_worker_runs"
    )
    source = sourced(captured(state_dir(p)), "parent.delegate")[-1]
    selected = json.loads(
        p.cli(
            "decide",
            "--spec",
            p.spec(
                {
                    "inputs": [source["id"]],
                    "evaluation": "investigate",
                    "reason": "Start the assigned worker",
                    "next_actions": [
                        {
                            "operation": "start_ai",
                            "summary": "Run the assigned work",
                            "actor": "worker",
                            "target": "child 10",
                            "instruction": "Use the saved assignment",
                            "conditions": "Saved execution basis",
                            "expected_result": "Worker result",
                        }
                    ],
                }
            ),
        ).stdout
    )
    injection = """
from decision_logs.capture import Journal
from worker_logs.journal import PublicationError
def broken_journal_read(self):
    raise PublicationError('journal_unavailable')
Journal.read = broken_journal_read
"""
    monkeypatch.setattr(
        worker_tests,
        "ENTRY",
        worker_tests.ENTRY.replace(
            "from parent.cli import main", injection + "\nfrom parent.cli import main"
        ),
    )
    response = run(p, "--decision-id", selected["decision_id"])
    assert response.returncode == 0
    assert worker_tests.task(p)["worker_runs"] and worker_tests.task(p)["worker_ended"]
    view = json.loads(p.cli("decision-log-status", ok=False).stdout)
    assert view["status"] == "incomplete" and view["failures"]


def test_invalid_worker_input_stays_rejected_when_rejection_capture_fails(
    reserved: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_tests = importlib.import_module(
        "tests.e2e.agents_ops.test_manual_worker_runs"
    )
    injection = """
from contextlib import contextmanager
from decision_logs import capture as capture_module
saved_lock = capture_module.locked
@contextmanager
def broken_lock(path):
    if path.parent.name == 'captures':
        raise OSError('injected rejection capture lock failure')
    with saved_lock(path):
        yield
capture_module.locked = broken_lock
"""
    monkeypatch.setattr(
        worker_tests,
        "ENTRY",
        worker_tests.ENTRY.replace(
            "from parent.cli import main", injection + "\nfrom parent.cli import main"
        ),
    )
    response = run(reserved, "--decision-id", "invalid", ok=False)
    assert response.returncode == 2 and "invalid_public_key" in response.stderr
    assert not (reserved.root / "runner-calls.jsonl").exists()
    view = json.loads(reserved.cli("decision-log-status", ok=False).stdout)
    assert view["status"] == "incomplete" and view["failures"]


def test_review_publication_failure_keeps_observed_result_and_new_run_predecessor(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    failed = json.loads(
        r.run(
            ok=False, extra={"PARENT_FAIL_PATH": "repos/o/r/issues/100/comments"}
        ).stdout
    )
    assert failed["publication"] == "publication_failed"
    rows = captured(state_dir(r.p))
    assert any(
        v["data"]["status"] == "success"
        and "review.result.json:" in v["data"]["summary"]
        for v in sourced(rows, "review.run")
    )
    assert any(
        "publication_failed" in v["data"]["summary"]
        for v in sourced(rows, "review.run")
    )
    assert json.loads(r.run(ok=False).stdout)["reused"]
    assert len(r.calls()) == 1
    manifest = json.loads(
        r.p.cli(
            "review-prepare",
            "--prs",
            "100",
            "--reason-file",
            r.p.note("公開済みの有効結果が得られず、新規レビューが必要。"),
        ).stdout
    )
    request = json.loads(
        r.p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            r.p.note("前runの取得結果を引き継いで確認する。"),
        ).stdout
    )
    r.request_path.write_text(json.dumps(request))
    r.run()
    rows = captured(state_dir(r.p))
    exceptions = [
        v
        for v in sourced(rows, "review.run")
        if "exception=True" in v["data"]["summary"]
    ]
    assert exceptions and "previous_review_keys=['" in exceptions[-1]["data"]["summary"]
    assert len(r.calls()) == 2
    assert all(
        s["parent_confirmation"] == "unrecorded" for s in graph.states(rows).values()
    )
