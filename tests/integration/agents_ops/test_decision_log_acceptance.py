"""Real capture/decision/journal/projection/navigation, with synthetic work facts.

No worker or network is started: overlapping operations model two independent
workers, while the actual journal and parent decision entry preserve causality.
"""

from __future__ import annotations

import html
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from tests.support.agents_ops.decision_logs import evidence, observed, schema

C: Any = importlib.import_module("decision_logs.capture")
PARENT: Any = importlib.import_module("decision_logs.parent_commands")
navigation: Any = importlib.import_module("decision_logs.navigation")


def test_delegation_direct_verification_parallel_join_and_handoff(
    tmp_path: Path,
) -> None:
    session = SimpleNamespace(
        owner=lambda: None,
        store=SimpleNamespace(directory=tmp_path / "state"),
        gh=SimpleNamespace(repo="o/r"),
        parent=1,
        session="codex:fixture-parent-before",
    )

    def decide(inputs: list[str], reason: str, operations: list[str]) -> dict[str, Any]:
        spec = {
            "inputs": inputs,
            "basis": [] if inputs else [evidence()],
            "evaluation": "continue",
            "reason": reason,
            "next_actions": [
                {
                    "operation": operation,
                    "summary": f"{operation}: follow the recorded plan",
                    "actor": "worker" if operation == "start_ai" else "parent",
                    "target": f"work item {index}",
                    "instruction": "Keep independent evidence and verify before acceptance",
                    "conditions": "No implicit acceptance from runtime completion",
                    "expected_result": "An explicit result",
                }
                for index, operation in enumerate(operations)
            ],
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        return cast(dict[str, Any], PARENT.decide(session, path))

    initial = decide(
        [], "Delegate two independent investigations", ["start_ai", "start_ai"]
    )
    workers = []
    for child, plan_id in zip((10, 20), initial["plan_ids"], strict=True):
        scope = {
            "repo": "o/r",
            "parent": 1,
            "issue": child,
            "activity_key": schema.new_id(),
            "delegation": {"key": "d-" + schema.new_id(), "turn": 1},
            "review_key": None,
        }
        worker = C.Operation.begin(
            session.store.directory,
            "o/r",
            1,
            "parent.run-worker",
            "start_ai",
            scope=scope,
            decision_id=initial["decision_id"],
            plan_id=plan_id,
        )
        worker.action("running", f"Worker for child {child} started")
        workers.append(worker)
    # Both starts precede both reports. Finishing in reverse order must not
    # turn the parallel branches into a causal dependency on each other.
    for worker in reversed(workers):
        worker.finish(
            {"findings": worker.row["scope"]["issue"]},
            status="partial",
            summary=f"Worker {worker.row['scope']['issue']} found a distinct issue",
            observer="worker_report",
            final_report=observed(evidence()),
        )
    report_ids = [w.records("result")[-1]["id"] for w in workers]
    direct = C.Operation.begin(
        session.store.directory, "o/r", 1, "parent.record-result", "verify"
    )
    direct.action("running", "Parent checks both artifacts directly")
    direct.finish(
        {"checked": [10, 20]},
        summary="Parent verified both independent artifacts",
        observer="parent_verification",
        verifies=report_ids,
    )
    joined = decide(
        [*report_ids, direct.records("result")[-1]["id"]],
        "Join the two findings and hand off remaining CI verification",
        ["handoff"],
    )
    handoff = C.Operation.begin(
        session.store.directory,
        "o/r",
        1,
        "parent.handoff",
        "handoff",
        decision_id=joined["decision_id"],
        plan_id=joined["plan_ids"][0],
    )
    handoff.finish(
        {"remaining": "CI verification"},
        summary="Handoff preserved both workers and pending CI evidence",
    )
    session.session = "codex:fixture-parent-after"
    final = decide(
        [handoff.records("result")[-1]["id"]],
        "New parent retains the saved evidence and will check CI",
        ["verify"],
    )
    log = direct.journal
    original = log.path.read_bytes()
    preview = log.preview_publication()
    assert preview["blocked"] == [] and log.path.read_bytes() == original
    public = {
        navigation.event_path(p["record"]): schema.json_bytes(p)
        for p in preview["payloads"]
    }
    generated = navigation.generate(public)
    navigation.validate_links({**public, **generated}, generated)
    text = html.unescape("\n".join(raw.decode() for raw in generated.values()))
    assert "Join the two findings" in text and "New parent retains" in text
    assert "判断済み未実行" in text and "引継ぎ" in text
    assert "親の直接活動" in text and "担当 #10" in text and "担当 #20" in text
    assert all(f"#record-{key}" in text for key in report_ids)
    rows = {p["record"]["id"]: p["record"] for p in preview["payloads"]}
    assert (
        rows[initial["decision_id"]]["data"]["source"]["actor_key"]
        == rows[joined["decision_id"]]["data"]["source"]["actor_key"]
    )
    assert (
        rows[final["decision_id"]]["data"]["source"]["actor_key"]
        != rows[joined["decision_id"]]["data"]["source"]["actor_key"]
    )
    for worker in workers:
        report = worker.records("result")[-1]
        assert report["data"]["previous_result_id"] is None
        assert report["data"]["verifies"] == []
    assert "fixture-parent-before" not in text and "fixture-parent-after" not in text
