"""Private receipt capture through the real journal and public-only renderer."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from tests.support.agents_ops.decision_logs import AT, evidence, journal, schema
from tests.support.agents_ops.traceability import termination_observation
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
    receipt = {
        k: v
        for k, v in termination_observation().items()
        if k
        in {
            "generation",
            "event",
            "observed_at",
            "main_revision",
            "remaining",
            "publication_actor",
            "resume_command",
        }
    }
    receipt.update(
        repo="o/r",
        parent=1,
        publication_actor="parent #1",
        resume_command="python -m parent --parent 1 publish-decision-log --retry",
        remaining=[
            {
                "kind": "decision_log",
                "parent": 1,
                "state": "pending",
                "stage": "pr_open",
                "error": None,
                "pr_number": 77,
                "pr_head": "c" * 40,
                "unpublished_count": 2,
                "batch_key": "b" * 32,
            }
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
    receipt = {
        "repo": "o/r",
        "parent": 1,
        "generation": 2,
        "event": "handoff_accepted",
        "observed_at": AT,
        "main_revision": None,
        "remaining": [],
        "publication_actor": "receiving parent",
        "resume_command": None,
    }
    path = tmp_path / "termination.json"
    worker_journal.atomic(path, receipt)
    trace.capture_termination(parent, path)
    public = public_files(parent.journal)
    page = navigation.generate(public)[
        ".agents/logs/parents/parent-1/README.md"
    ].decode()
    assert "引継ぎ受領" in page and "unverified" in page
    assert "completed" not in page


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
    receipt = {
        "repo": "o/r",
        "parent": 1,
        "generation": 1,
        "event": "ended",
        "observed_at": AT,
        "main_revision": None,
        "remaining": [
            {
                "kind": "worker_log",
                "parent": 1,
                "issue": 10,
                "public_key": "d-" + "a" * 32,
                "sequence": 2,
                "state": "pending",
                "stage": "mergeability_pending",
                "error": None,
                "pr_number": 42,
                "pr_head": "d" * 40,
            }
        ],
        "publication_actor": "parent #1900",
        "resume_command": "python -m parent --parent 1 publish-worker-log --child 10 --delegation-key d-"
        + "a" * 32
        + " --turn 2",
    }
    trace.capture_termination(parent, receipt)
    page = navigation.generate(public_files(parent.journal))[
        ".agents/logs/parents/parent-1/README.md"
    ].decode()
    assert "https://github.com/o/r/pull/42" in page
    assert "mergeability" in page and "--turn 2" in page and "1900" in page
    assert "d-" + "a" * 32 in page
