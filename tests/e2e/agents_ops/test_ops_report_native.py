"""Native evidence from anonymized Codex 0.163.0 execution records; no live CLI."""

from __future__ import annotations

import copy
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report"))
native: Any = importlib.import_module("opsreport.native")
FIXTURE = Path(__file__).parent / "fixtures/native_transcripts.json"


@dataclass
class Records:
    root: Path
    data: dict[str, Any]

    @property
    def run(self) -> Path:
        return self.root / "agent-runtime"

    @property
    def home(self) -> Path:
        return self.root / "codex-home"

    def path(self, child: str | None = None) -> Path:
        session = f"native-{child}" if child is not None else "parent-session"
        return self.home / "sessions" / f"rollout-{session}.jsonl"

    def save(self) -> None:
        (self.run / "result.json").write_text(json.dumps(self.data["result"]))
        self.write(self.path(), self.data["parent"])
        for name, rows in self.data["children"].items():
            self.write(self.path(name), rows)

    def write(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def tasks(self) -> dict[str, str]:
        return {f"question-{name}": f"native-{name}" for name in self.data["children"]}


@pytest.fixture
def records(tmp_path: Path) -> Records:
    root = tmp_path
    raw = FIXTURE.read_text()
    # Only filesystem placeholders are expanded; native names/IDs stay symbolic.
    data = json.loads(
        raw.replace('"CODEX_HOME"', json.dumps(str(root / "codex-home"))).replace(
            '"WORKTREE"', json.dumps(str(root / "worktree"))
        )
    )
    fixture = Records(root, data)
    fixture.run.mkdir()
    fixture.save()
    return fixture


def _row(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    return next(row for row in rows if row["payload"].get("type", row["type"]) == kind)


def test_real_record_shape_proves_overlap_rejection_and_followup(
    records: Records,
) -> None:
    evidence = native.inspect_run(records.run)
    assert evidence.parent_session_id == "parent-session"
    assert evidence.parent_turn_id == "parent-turn"
    assert [a.task_name for a in evidence.agents] == [
        "/root/a",
        "/root/b",
        "/root/c",
        "/root/after_completion",
    ]
    a, b, c, followup = evidence.agents
    assert max(a.started_at, b.started_at, c.started_at) < min(
        a.completed_at, b.completed_at, c.completed_at
    )
    assert a.completed_at < followup.started_at < b.completed_at
    assert all(agent.status == "completed" for agent in evidence.agents)
    assert all(
        agent.model == "gpt-6-astra" and agent.effort == "max"
        for agent in evidence.agents
    )
    assert all(
        agent.sandbox_policy == {"type": "read-only"} for agent in evidence.agents
    )
    assert len(evidence.rejections) == 1
    assert evidence.rejections[0].task_name == "d"
    assert (
        evidence.rejections[0].error
        == "collab spawn failed: agent thread limit reached"
    )
    assert re.fullmatch("[a-f0-9]{64}", evidence.transcript.sha256)
    matched = native.reconcile(records.run, records.tasks())
    assert matched["question-a"].agent_id == "native-a"
    assert matched["question-a"].spawn_call_id == "spawn-a"
    assert matched["question-a"].result == "PROBE_A_DONE"


def test_agent_self_report_is_not_an_execution_source(records: Records) -> None:
    (records.run / "final.md").write_text('{"all_children_completed": true}')
    records.path("a").unlink()
    with pytest.raises(native.NativeEvidenceError, match="transcript missing"):
        native.reconcile(records.run, records.tasks())


@pytest.mark.parametrize(
    "mapping",
    [
        {"q": "native-missing"},
        {"q": "/root/a"},
        {"q": "native-a"},
        {"q": "native-a", "other": "native-a"},
        {"q": ""},
        {"": "native-a"},
        {"q": "native-d"},
    ],
)
def test_ledger_ids_must_form_a_bijection(
    records: Records, mapping: dict[str, str]
) -> None:
    with pytest.raises(native.NativeEvidenceError, match="ledger"):
        native.reconcile(records.run, mapping)


@pytest.mark.parametrize("aborted", [False, True])
def test_incomplete_child_is_visible_but_cannot_verify(
    records: Records, aborted: bool
) -> None:
    rows = records.data["children"]["a"]
    finish = rows.pop()
    assert finish["payload"]["type"] == "task_complete"
    if aborted:
        rows.append(
            {**finish, "payload": {"type": "turn_aborted", "turn_id": "turn-a"}}
        )
    records.save()
    agent = native.inspect_run(records.run).agents[0]
    assert agent.status == ("aborted" if aborted else "incomplete")
    assert agent.completed_at is None and agent.result is None
    with pytest.raises(native.NativeEvidenceError, match="not completed"):
        native.reconcile(records.run, records.tasks())


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other-model"),
        ("effort", "low"),
        ("sandbox_policy", {"type": "danger-full-access"}),
        ("cwd", "foreign-worktree"),
        ("approval_policy", "on-request"),
    ],
)
def test_child_inherited_settings_are_observed(
    records: Records, field: str, value: Any
) -> None:
    _row(records.data["children"]["a"], "turn_context")["payload"][field] = value
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="inherited"):
        native.inspect_run(records.run)


@pytest.mark.parametrize("kind", ["task_started", "turn_context", "message"])
def test_missing_child_evidence_is_not_inferred(records: Records, kind: str) -> None:
    rows = records.data["children"]["a"]
    rows.remove(_row(rows, kind))
    records.save()
    with pytest.raises(native.NativeEvidenceError):
        native.inspect_run(records.run)


@pytest.mark.parametrize(
    "kind", ["session_meta", "task_started", "turn_context", "task_complete"]
)
def test_duplicate_child_evidence_is_rejected(records: Records, kind: str) -> None:
    rows = records.data["children"]["a"]
    row = _row(rows, kind)
    rows.insert(rows.index(row), copy.deepcopy(row))
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="duplicate|ambiguous"):
        native.inspect_run(records.run)


@pytest.mark.parametrize("kind", ["function_call", "function_call_output"])
def test_duplicate_parent_tool_record_is_rejected(records: Records, kind: str) -> None:
    rows = records.data["parent"]
    rows.insert(4, copy.deepcopy(_row(rows, kind)))
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="duplicate tool"):
        native.inspect_run(records.run)


@pytest.mark.parametrize("kind", ["function_call", "function_call_output"])
def test_missing_spawn_call_or_receipt_is_rejected(records: Records, kind: str) -> None:
    rows = records.data["parent"]
    rows.remove(_row(rows, kind))
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="matching call|unpaired"):
        native.inspect_run(records.run)


@pytest.mark.parametrize(
    "output",
    [
        '{"task_name":"/root/a","task_name":"/root/b"}',
        '{"task_name":',
        '{"task_name":"/root/wrong"}',
        "{}",
        "unknown failure",
    ],
)
def test_broken_spawn_receipt_is_not_a_rejection(records: Records, output: str) -> None:
    _row(records.data["parent"], "function_call_output")["payload"]["output"] = output
    records.save()
    with pytest.raises(native.NativeEvidenceError):
        native.inspect_run(records.run)


@pytest.mark.parametrize("target", ["parent", "child", "result"])
def test_corrupt_record_is_an_explicit_error(records: Records, target: str) -> None:
    path = (
        records.run / "result.json"
        if target == "result"
        else records.path("a" if target == "child" else None)
    )
    path.write_text('{"duplicate":1,"duplicate":2}\n')
    with pytest.raises(native.NativeEvidenceError, match="duplicate key"):
        native.inspect_run(records.run)


def test_partial_final_line_is_not_silently_skipped(records: Records) -> None:
    path = records.path("a")
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(native.NativeEvidenceError, match="truncated"):
        native.inspect_run(records.run)


@pytest.mark.parametrize("child", [None, "a"])
def test_live_and_archived_duplicate_is_ambiguous(
    records: Records, child: str | None
) -> None:
    original = records.path(child)
    archive = records.home / "archived_sessions" / original.name
    archive.parent.mkdir()
    archive.write_bytes(original.read_bytes())
    with pytest.raises(native.NativeEvidenceError, match="ambiguous"):
        native.inspect_run(records.run)


def test_archived_child_is_a_valid_source(records: Records) -> None:
    original = records.path("a")
    archive = records.home / "archived_sessions" / original.name
    archive.parent.mkdir()
    original.rename(archive)
    assert native.reconcile(records.run, records.tasks())[
        "question-a"
    ].transcript.path == str(archive)


@pytest.mark.parametrize(
    "field", ["id", "session_id", "parent_thread_id", "agent_path"]
)
def test_foreign_child_session_is_rejected(records: Records, field: str) -> None:
    records.data["children"]["a"][0]["payload"][field] = "foreign-session"
    records.save()
    with pytest.raises(native.NativeEvidenceError):
        native.inspect_run(records.run)


@pytest.mark.parametrize("where", ["spawn", "start", "finish", "context", "tool"])
def test_foreign_parent_or_turn_is_rejected(records: Records, where: str) -> None:
    rows = records.data["children"]["a"]
    if where == "spawn":
        rows[0]["payload"]["source"]["subagent"]["thread_spawn"]["parent_thread_id"] = (
            "other-parent"
        )
    elif where == "start":
        _row(rows, "task_started")["payload"]["turn_attribution"]["parent_turn_id"] = (
            "other-turn"
        )
    elif where == "finish":
        _row(rows, "task_complete")["payload"]["turn_id"] = "other-turn"
    elif where == "context":
        _row(rows, "turn_context")["payload"]["turn_id"] = "other-turn"
    else:
        _row(records.data["parent"], "function_call")["payload"][
            "internal_chat_message_metadata_passthrough"
        ]["turn_id"] = "other-turn"
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="mismatch|turn"):
        native.inspect_run(records.run)


def test_final_message_and_completion_must_agree(records: Records) -> None:
    _row(records.data["children"]["a"], "task_complete")["payload"][
        "last_agent_message"
    ] = "invented result"
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="final message disagree"):
        native.inspect_run(records.run)


def test_successful_runner_requires_parent_completion(records: Records) -> None:
    records.data["parent"].pop()
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="completion mismatch"):
        native.inspect_run(records.run)


def test_reusing_a_child_cannot_reuse_its_initial_completion(records: Records) -> None:
    call = copy.deepcopy(_row(records.data["parent"], "function_call"))
    call["payload"].update(name="followup_task", call_id="followup-a")
    records.data["parent"].insert(-1, call)
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="followup_task"):
        native.reconcile(records.run, records.tasks())


def test_other_parent_session_is_not_reused(records: Records) -> None:
    meta = records.data["children"]["a"][0]["payload"]
    meta["parent_thread_id"] = "foreign-parent"
    meta["source"]["subagent"]["thread_spawn"]["parent_thread_id"] = "foreign-parent"
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="transcript missing"):
        native.inspect_run(records.run)


def test_extra_child_with_no_recorded_spawn_is_an_error(records: Records) -> None:
    rows = json.loads(
        json.dumps(records.data["children"]["a"])
        .replace("native-a", "native-extra")
        .replace("/root/a", "/root/extra")
    )
    records.write(records.path("extra"), rows)
    with pytest.raises(native.NativeEvidenceError, match="no matching spawn"):
        native.inspect_run(records.run)


def test_resume_uses_recorded_turn_not_latest_session_context(records: Records) -> None:
    # Later parent/child turns must not replace this run's starts or results.
    for rows in [records.data["parent"], records.data["children"]["a"]]:
        start = copy.deepcopy(_row(rows, "task_started"))
        context = copy.deepcopy(_row(rows, "turn_context"))
        for row in (start, context):
            row["payload"]["turn_id"] = "later-turn"
        rows.extend([start, context])
    records.save()
    assert (
        native.reconcile(records.run, records.tasks())["question-a"].result
        == "PROBE_A_DONE"
    )


def test_child_from_another_runner_turn_is_not_added(records: Records) -> None:
    rows = json.loads(
        json.dumps(records.data["children"]["a"])
        .replace("native-a", "native-previous")
        .replace("/root/a", "/root/previous")
        .replace("parent-turn", "previous-parent-turn")
    )
    records.write(records.path("previous"), rows)
    assert len(native.inspect_run(records.run).agents) == 4


@pytest.mark.parametrize("field", ["session_id", "turn_id"])
def test_missing_runner_identity_is_not_discovered_from_environment(
    records: Records, field: str
) -> None:
    records.data["result"]["runtime"].pop(field)
    records.save()
    with pytest.raises(native.NativeEvidenceError, match="identifier"):
        native.inspect_run(records.run)


def test_missing_runner_record_is_an_error(records: Records) -> None:
    (records.run / "result.json").unlink()
    with pytest.raises(native.NativeEvidenceError):
        native.inspect_run(records.run)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 99),
        ("mode", "write"),
        ("agent", "claude"),
        ("model", "unrequested-model"),
        ("runtime_home", "relative-home"),
    ],
)
def test_invalid_runner_contract_is_rejected(
    records: Records, field: str, value: Any
) -> None:
    result = records.data["result"]
    if field == "schema_version":
        result[field] = value
    elif field == "agent":
        result["contract"]["settings"][field] = value
    elif field == "model":
        result["contract"]["settings"]["selected"][field] = value
    else:
        result["contract"][field] = value
    records.save()
    with pytest.raises(native.NativeEvidenceError):
        native.inspect_run(records.run)


@pytest.mark.parametrize("field,value", [("fork_turns", "all"), ("namespace", "other")])
def test_unverified_native_launch_shape_is_not_accepted(
    records: Records, field: str, value: str
) -> None:
    call = _row(records.data["parent"], "function_call")["payload"]
    if field == "namespace":
        call[field] = value
    else:
        args = json.loads(call["arguments"])
        args[field] = value
        call["arguments"] = json.dumps(args)
    records.save()
    with pytest.raises(native.NativeEvidenceError):
        native.inspect_run(records.run)


def test_fixture_has_no_real_session_identifiers_or_filesystem_paths() -> None:
    text = FIXTURE.read_text()
    assert not re.search(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", text)
    assert "/home/" not in text and "/mnt/" not in text
    assert "gAAAAA" not in text
    data = json.loads(text)
    assert data["result"]["contract"]["cwd"] == "WORKTREE"
    assert data["result"]["contract"]["runtime_home"] == "CODEX_HOME"
