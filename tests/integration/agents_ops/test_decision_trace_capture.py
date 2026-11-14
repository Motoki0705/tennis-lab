"""Private receipt capture through the real journal and public-only renderer."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import evidence, journal, schema
from tests.support.agents_ops.traceability import (
    lifecycle_receipt,
    remaining_publication,
)
from tests.support.agents_ops.worker_logs import record_source

capture: Any = importlib.import_module("decision_logs.capture")
observations: Any = importlib.import_module("decision_logs.observations")
trace: Any = importlib.import_module("decision_logs.trace_capture")
worker_journal: Any = importlib.import_module("worker_logs.journal")
worker_sources: Any = importlib.import_module("worker_logs.sources")
navigation: Any = importlib.import_module("decision_logs.navigation")
historical: Any = importlib.import_module("decision_logs.historical")


def public_files(log: Any) -> dict[str, bytes]:
    preview = log.preview_publication()
    assert not preview["blocked"], preview
    return {
        navigation.event_path(p["record"]): schema.json_bytes(p)
        for p in preview["payloads"]
    }


@pytest.mark.parametrize("inventory", ["complete", "partial", "unavailable"])
def test_termination_preserves_unknown_counts_and_known_publications(
    tmp_path: Path, inventory: str
) -> None:
    """PR #1984's capture failure must retain both known work and unknowns."""
    parent = capture.Operation.begin(tmp_path, "o/r", 1, "parent.release", "handoff")
    remaining = [
        remaining_publication("worker_log"),
        remaining_publication(
            "decision_log",
            state="captured",
            stage="capture_incomplete",
            error="input_capture_failed",
            unpublished_count=None,
        ),
    ]
    receipt = lifecycle_receipt(
        remaining=remaining if inventory != "unavailable" else [],
        remaining_complete=inventory == "complete",
        error="journal_unavailable"
        if inventory == "unavailable"
        else "input_capture_failed",
        publication_actor="unassigned" if inventory == "unavailable" else 1900,
    )
    original = schema.json_bytes(receipt)
    path = tmp_path / "exit.json"
    worker_journal.atomic(path, receipt)
    before = path.read_bytes()
    first = trace.capture_termination(parent, path)
    assert trace.capture_termination(parent, receipt) == first
    files = public_files(parent.journal)
    result = next(r for r in navigation.records(files) if r["kind"] == "result")
    fact = result["data"]["observation"]
    assert result["data"]["status"] == "partial"
    assert fact["remaining"] == receipt["remaining"]
    assert fact["remaining_complete"] is receipt["remaining_complete"]
    assert fact["publication_actor"] == receipt["publication_actor"]
    assert fact["error"] == receipt["error"]
    assert fact["decision_resume_command"] == receipt["decision_resume_command"]
    text = navigation.generate(files)[
        ".agents/logs/parents/parent-1/README.md"
    ].decode()
    assert "publish-worker-log" in text and "publish-decision-log" in text
    assert "公開残件（観測時点）: 0 件" not in text
    if inventory == "unavailable":
        assert "公開残件の照合: 未確認" in text and "journal" in text
    else:
        assert "https://github.com/o/r/pull/42" in text
        assert "未公開記録数: 未確認" in text
        assert "未公開 0 記録" not in text and "未公開 None" not in text
        assert "input" in text and "1900" in text
        if inventory == "partial":
            assert "公開残件の照合: 未確認" in text
    assert path.read_bytes() == before and schema.json_bytes(receipt) == original
    assert navigation.generate(
        {**files, **navigation.generate(files)}
    ) == navigation.generate(files)


def test_failed_worker_with_a_final_is_observed_without_publication_success(
    tmp_path: Path,
) -> None:
    source, receipt = record_source()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source["run_directory"] = str(runtime)
    record = receipt["record"]
    record.update(result="failed", turn_end_reason="runtime_error", agent_spawned=True)
    worker_journal.atomic(runtime / "result.json", record)
    (runtime / "final.md").write_text(receipt["final"])
    path = worker_sources.location(tmp_path, "o/r", 1, 10, "a" * 32, 1)
    worker_journal.atomic(
        path,
        {
            "version": 1,
            "source": source,
            "state": "worker_failed",
            "stage": "awaiting_final",
            "error": "worker_failed",
        },
    )
    before = path.read_bytes()
    observations.worker_log(
        path,
        {"state": "worker_failed", "saved_to_main": False, "error": "worker_failed"},
    )
    log = journal.Journal(tmp_path, "o/r", 1)
    public = public_files(log)
    rows = navigation.records(public)
    fact = next(r for r in rows if r["kind"] == "result")["data"]
    assert fact["final_report"]["status"] == "observed"
    assert fact["observation"]["execution"]["state"] == "ended"
    assert fact["observation"]["publication"]["state"] == "pending"
    assert fact["status"] == "failure"
    text = "\n".join(v.decode() for v in navigation.generate(public).values())
    assert "最終報告あり・公開保留" in text
    assert "private-session-value" not in text and receipt["final"] not in text
    assert path.read_bytes() == before


def test_termination_after_publication_then_reconciliation_preserves_both_observations(
    tmp_path: Path,
) -> None:
    log = journal.Journal(tmp_path, "o/r", 1)
    # A prior publication contains no lifecycle facts. The next capture must
    # expose the later end, without rewriting the earlier public files.
    op = capture.Operation.begin(tmp_path, "o/r", 1, "fixture.prior", "record")
    op.finish({"published": True})
    before = public_files(log)
    parent = capture.Operation.begin(tmp_path, "o/r", 1, "parent.release", "handoff")
    receipt = lifecycle_receipt(
        publication_actor=1,
        decision_resume_command="python -m parent --parent 1 publish-decision-log --retry",
        remaining=[
            remaining_publication(
                "decision_log",
                state="pending",
                stage="pr_open",
                pr_number=77,
                pr_head="c" * 40,
                public_key="b" * 32,
                batch_key="b" * 32,
            )
        ],
    )
    first = trace.capture_termination(
        parent,
        receipt,
        evidence=[evidence()],
        acceptance={"state": "completed", "evidence": [evidence()]},
    )
    assert (
        trace.capture_termination(
            parent,
            receipt,
            evidence=[evidence()],
            acceptance={"state": "completed", "evidence": [evidence()]},
        )
        == first
    )
    pending = public_files(log)
    parent_page = navigation.generate(pending)[
        ".agents/logs/parents/parent-1/README.md"
    ].decode()
    assert (
        "担当終了" in parent_page
        and "未公開 2 記録" in parent_page
        and "publish-decision-log" in parent_page
    )
    receipt.update(observed_at="2026-11-10T09:00:00+00:00", remaining=[])
    second = trace.capture_termination(
        parent,
        receipt,
        evidence=[evidence()],
        acceptance={"state": "completed", "evidence": [evidence()]},
    )
    assert second != first
    after = public_files(log)
    assert all(after[name] == raw for name, raw in before.items())
    generated = navigation.generate(after)
    page = generated[".agents/logs/parents/parent-1/README.md"].decode()
    assert "公開残件（観測時点）: 0 件" in page
    assert first.encode() in b"\n".join(generated.values())
    assert navigation.generate({**after, **generated}) == generated


def test_handoff_and_unverified_acceptance_are_not_completed_by_runtime_end(
    tmp_path: Path,
) -> None:
    parent = capture.Operation.begin(tmp_path, "o/r", 1, "parent.handoff", "handoff")
    receipt = lifecycle_receipt(
        generation=2,
        event="handoff_accepted",
        main_revision=None,
        publication_actor=2,
    )
    path = tmp_path / "termination.json"
    worker_journal.atomic(path, receipt)
    first = trace.capture_termination(parent, path)
    public = public_files(parent.journal)
    page = navigation.generate(public)[
        ".agents/logs/parents/parent-1/README.md"
    ].decode()
    assert "引継ぎ受領" in page and "unverified" in page
    assert "completed" not in page
    # Acceptance evidence may become available after the lifecycle receipt.
    # Re-reading that same receipt must preserve the newly explicit judgment.
    second = trace.capture_termination(
        parent,
        path,
        acceptance={"state": "completed", "evidence": [evidence()]},
    )
    assert second != first
    records = navigation.records(public_files(parent.journal))
    assert [
        row["data"]["observation"]["acceptance"]["state"]
        for row in records
        if row["kind"] == "result"
    ] == ["unverified", "completed"]
    latest = navigation.generate(public_files(parent.journal))[
        ".agents/logs/parents/parent-1/README.md"
    ].decode()
    assert "最後の親の受入（明示記録時点）: completed" in latest


def test_historical_import_is_explicit_idempotent_and_namespace_bounded(
    tmp_path: Path,
) -> None:
    log = journal.Journal(tmp_path, "Motoki0705/tennis-lab", 1796)
    first = historical.capture(log)
    original = log.path.read_bytes()
    assert len(first) == 4 and historical.capture(log) == first
    assert log.path.read_bytes() == original
    assert not log.path.with_name("publication.json").exists()
    assert all(
        r["scope"]["parent"] == 1796 for r in navigation.records(public_files(log))
    )


def test_termination_keeps_worker_publication_identity_and_resume_destination(
    tmp_path: Path,
) -> None:
    parent = capture.Operation.begin(tmp_path, "o/r", 1, "parent.release", "handoff")
    receipt = lifecycle_receipt(
        main_revision=None,
        remaining=[remaining_publication("worker_log", stage="mergeability_pending")],
        publication_actor=1900,
        resume_command="python -m parent --parent 1 publish-worker-log --child 10 --delegation-key d-"
        + "a" * 32
        + " --turn 2",
    )
    trace.capture_termination(parent, receipt)
    page = navigation.generate(public_files(parent.journal))[
        ".agents/logs/parents/parent-1/README.md"
    ].decode()
    assert "https://github.com/o/r/pull/42" in page
    assert "mergeability" in page and "--turn 2" in page and "1900" in page
    assert "d-" + "a" * 32 in page
