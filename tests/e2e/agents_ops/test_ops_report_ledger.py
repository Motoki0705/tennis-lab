"""Audit history survives interruption without losing provenance or open questions."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report"))
ledger: Any = importlib.import_module("opsreport.ledger")
BASE = "a" * 40
SLOT = "20261024"
START = "2026-10-24T00:00:00+00:00"
AT = "2026-10-24T01:00:00+00:00"
GAP = {
    "scope": "src/legacy",
    "reason": "native tool unavailable",
    "resume": "restore tool, then retry this task",
}


def _task(
    key: str,
    *,
    origin: dict[str, str] | None = None,
    question: str | None = None,
    scope: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": key,
        "question": question or f"What is the impact of {key}?",
        "scope": scope or [f"src/{key}"],
        "origin": origin or {"kind": "goal", "id": SLOT},
        "agent_id": None,
        "status": "planned",
        "started_at": None,
        "finished_at": None,
        "conclusion": "",
        "evidence": [],
        "next_questions": [],
        "unconfirmed": [],
    }


def _evidence(key: str = "e-code", **changes: Any) -> dict[str, Any]:
    return {
        "id": key,
        "source_kind": "code",
        "reference": "src/legacy.py:10",
        "acquired_at": START,
        "target_version": BASE,
        "pr_head": None,
        **changes,
    }


def _finding(
    key: str = "f-root",
    *,
    task_id: str = "t-root",
    kind: str = "design",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": key,
        "task_id": task_id,
        "kind": kind,
        "status": "suspected",
        "conclusion": "A candidate requires verification",
        "evidence": [],
        "counterevidence": [],
        "details": details or {},
    }


def _paths(*, used: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "outcome": "used" if kind == used else "unused",
            "evidence": ["e-code"],
            "reason": "entrypoint exercised"
            if kind == used
            else "entrypoint inventory checked at the pinned version",
        }
        for kind in sorted(ledger.USAGE_PATHS)
    ]


def _vulnerability() -> dict[str, Any]:
    return {
        "target_version": BASE,
        "conditions": ["untrusted path supplied to the CLI"],
        "reachability": "CLI passes the user path into the vulnerable dependency",
        "impact": "read files accessible to the process",
        "remediation": "validate paths before dependency calls",
        "verification": "isolated regression fixture with traversal input",
        "usage_paths": [_paths(used="config-cli")[0]],
    }


def _ending(status: str = "incomplete") -> dict[str, Any]:
    return {
        "status": status,
        "reason": "required tools are unavailable"
        if status == "incomplete"
        else "important hypotheses resolved within the stated scope",
        "checked_scope": ["src/legacy"],
        "excluded_scope": ["external submodule internals"],
        "unverified_risks": [GAP.copy()] if status == "incomplete" else [],
        "further_investigation": ["retry tool-dependent checks"]
        if status == "incomplete"
        else [],
    }


def _start(book: Any, key: str, *, retry: bool = False) -> None:
    row = book.snapshot["tasks"][key]
    row.update(
        status="running",
        agent_id=f"native-{key}{'-retry' if retry else ''}",
        started_at="2026-10-24T00:30:00+00:00"
        if retry
        else "2026-10-24T00:01:00+00:00",
        finished_at=None,
    )
    book.record_task(row, at=AT)


def _finish(
    book: Any, key: str, status: str = "completed", *, retry: bool = False
) -> None:
    row = book.snapshot["tasks"][key]
    row.update(
        status=status,
        finished_at="2026-10-24T00:40:00+00:00"
        if retry
        else "2026-10-24T00:02:00+00:00",
        conclusion="checked the execution paths"
        if status == "completed"
        else "tool failed; execution coverage is incomplete",
        evidence=["e-code"],
        unconfirmed=[] if status == "completed" else [GAP.copy()],
    )
    book.record_task(row, at=AT)


@pytest.fixture
def book(tmp_path: Path) -> Any:
    result = ledger.Ledger.create(
        tmp_path / "audit.json",
        slot=SLOT,
        base_sha=BASE,
        goal="Find evidence-backed improvements",
        at=START,
    )
    result.record_evidence(_evidence(), at=AT)
    return result


@pytest.fixture
def rooted(book: Any) -> Any:
    book.record_task(_task("t-root"), at=AT)
    return book


def test_variable_multistage_history_survives_interruption_and_resume(
    book: Any,
) -> None:
    for i in range(7):
        origin = (
            {"kind": "goal", "id": SLOT}
            if i in {0, 6}
            else {
                "kind": "finding" if i in {2, 4} else "task",
                "id": f"f-{i - 1}" if i in {2, 4} else f"t-{i - 1}",
            }
        )
        book.record_task(_task(f"t-{i}", origin=origin), at=AT)
        _start(book, f"t-{i}")
        _finish(book, f"t-{i}", "interrupted" if i == 6 else "completed")
        if i in {1, 3, 5}:
            finding = _finding(f"f-{i}", task_id=f"t-{i}")
            book.record_finding(finding, at=AT)
            if i != 5:
                finding.update(status="confirmed", evidence=["e-code"])
                book.record_finding(finding, at=AT)
    for i, kind in enumerate(("issue", "pr", "proposal", "unassigned")):
        book.record_handoff(
            {
                "id": f"h-{i}",
                "finding_id": "f-1",
                "kind": kind,
                "reference": None if kind == "unassigned" else f"record-{i}",
                "related_parents": [1349],
                "reason": "record the intended correction route; no action is authorized",
            },
            at=AT,
        )
    book.end(_ending(), at=AT)
    book.bind_report(1500, at=AT)
    snapshot = book.snapshot
    prefix = json.loads(book.path.read_text())["events"]
    restored = ledger.Ledger.load(book.path, slot=SLOT, base_sha=BASE)
    assert restored.snapshot == snapshot
    assert len(snapshot["tasks"]) == 7
    assert snapshot["tasks"]["t-4"]["origin"] == {"kind": "finding", "id": "f-3"}
    assert snapshot["findings"]["f-3"]["task_id"] == "t-3"
    summary = restored.summary()
    assert [r["id"] for r in summary["unfinished_tasks"]] == ["t-6"]
    assert [r["id"] for r in summary["unverified_hypotheses"]] == ["f-5"]
    interrupted_ref = summary["unfinished_tasks"][0]["ref"]
    restored.resume("tools restored; continue the same slot and report", at=AT)
    _start(restored, "t-6", retry=True)
    _finish(restored, "t-6", retry=True)
    finding = restored.snapshot["findings"]["f-5"]
    finding.update(status="confirmed", evidence=["e-code"])
    restored.record_finding(finding, at=AT)
    restored.end(_ending("completed"), at=AT)
    final = ledger.Ledger.load(book.path, slot=SLOT, base_sha=BASE)
    assert final.snapshot == restored.snapshot
    assert final.snapshot["run"]["report_issue"] == 1500
    assert [r["status"] for r in final.snapshot["run"]["endings"]] == [
        "incomplete",
        "completed",
    ]
    assert final.original(interrupted_ref)["status"] == "interrupted"
    assert final.snapshot["tasks"]["t-6"]["agent_id"] == "native-t-6-retry"
    assert json.loads(book.path.read_text())["events"][: len(prefix)] == prefix


@pytest.mark.parametrize(
    "origin",
    [
        {"kind": "task", "id": "missing"},
        {"kind": "finding", "id": "missing"},
        {"kind": "goal", "id": "another-slot"},
    ],
)
def test_missing_origins_are_explicit_errors(book: Any, origin: dict[str, str]) -> None:
    before = book.path.read_bytes()
    with pytest.raises(ledger.LedgerError, match="origin"):
        book.record_task(_task("t-invalid", origin=origin), at=AT)
    assert book.path.read_bytes() == before


@pytest.mark.parametrize("via_finding", [False, True])
def test_cycles_through_tasks_or_findings_are_rejected(
    rooted: Any, via_finding: bool
) -> None:
    book = rooted
    if via_finding:
        book.record_finding(_finding(), at=AT)
        origin = {"kind": "finding", "id": "f-root"}
    else:
        book.record_task(
            _task("t-child", origin={"kind": "task", "id": "t-root"}), at=AT
        )
        origin = {"kind": "task", "id": "t-child"}
    task = book.snapshot["tasks"]["t-root"]
    task["origin"] = origin
    with pytest.raises(ledger.LedgerError, match="origin cycle"):
        book.record_task(task, at=AT)


def test_self_origin_is_a_cycle(book: Any) -> None:
    with pytest.raises(ledger.LedgerError, match="origin cycle"):
        book.record_task(
            _task("t-self", origin={"kind": "task", "id": "t-self"}), at=AT
        )


@pytest.mark.parametrize(
    "first,last",
    [
        ("planned", "completed"),
        ("running", "planned"),
        ("completed", "running"),
        ("failed", "planned"),
        ("interrupted", "completed"),
        ("cancelled", "running"),
    ],
)
def test_illegal_task_state_transitions(rooted: Any, first: str, last: str) -> None:
    book = rooted
    if first != "planned":
        _start(book, "t-root")
        if first != "running":
            _finish(book, "t-root", first)
    task = book.snapshot["tasks"]["t-root"]
    task["status"] = last
    with pytest.raises(ledger.LedgerError, match="invalid transition"):
        book.record_task(task, at=AT)


def test_launch_failure_without_native_id_is_resumable(rooted: Any) -> None:
    task = rooted.snapshot["tasks"]["t-root"]
    task.update(
        status="failed",
        finished_at=AT,
        conclusion="native launch unavailable",
        unconfirmed=[GAP.copy()],
    )
    rooted.record_task(task, at=AT)
    restored = ledger.Ledger.load(rooted.path, slot=SLOT, base_sha=BASE)
    task = restored.snapshot["tasks"]["t-root"]
    task.update(
        status="running", agent_id="native-recovered", started_at=AT, finished_at=None
    )
    restored.record_task(task, at=AT)
    assert restored.snapshot["tasks"]["t-root"]["agent_id"] == "native-recovered"


def test_question_normalization_prevents_duplicate_launches(book: Any) -> None:
    book.record_task(
        _task(
            "t-a", question=" Code\n usage? ", scope=["src/a.py", " configs/cli.yaml "]
        ),
        at=AT,
    )
    with pytest.raises(ledger.LedgerError, match="duplicate question/scope.*t-a"):
        book.record_task(
            _task(
                "t-b",
                question="ＣＯＤＥ  usage?",
                scope=["configs/cli.yaml", "src/a.py"],
            ),
            at=AT,
        )
    book.record_task(
        _task("t-c", question="code usage?", scope=["src/another.py"]), at=AT
    )
    _start(book, "t-a")
    _finish(book, "t-a", "failed")
    with pytest.raises(ledger.LedgerError, match="duplicate question/scope"):
        book.record_task(
            _task(
                "t-d", question="code usage?", scope=["src/a.py", "configs/cli.yaml"]
            ),
            at=AT,
        )


@pytest.mark.parametrize("slot,base", [("20261025", BASE), (SLOT, "b" * 40)])
def test_load_rejects_a_different_slot_or_base(book: Any, slot: str, base: str) -> None:
    before = book.path.read_bytes()
    with pytest.raises(ledger.LedgerError, match="slot/base_sha mismatch"):
        ledger.Ledger.load(book.path, slot=slot, base_sha=base)
    assert book.path.read_bytes() == before


def test_pinned_code_evidence_cannot_mix_checkouts(book: Any) -> None:
    with pytest.raises(ledger.LedgerError, match="code base_sha mismatch"):
        book.record_evidence(_evidence("e-wrong", target_version="b" * 40), at=AT)


@pytest.mark.parametrize(
    "change,message",
    [
        ("unknown-evidence", "unknown evidence"),
        ("missing-agent", "agent_id"),
        ("no-timezone", "timezone"),
        ("backwards-finish", "finish interval"),
        ("missing-resume", "scope reason resume"),
    ],
)
def test_incomplete_or_unverifiable_task_results_stop(
    rooted: Any, change: str, message: str
) -> None:
    _start(rooted, "t-root")
    task = rooted.snapshot["tasks"]["t-root"]
    task.update(
        status="failed",
        finished_at=AT,
        conclusion="verification blocked",
        unconfirmed=[GAP.copy()],
    )
    if change == "unknown-evidence":
        task["evidence"] = ["missing"]
    elif change == "missing-agent":
        task["agent_id"] = None
    elif change == "no-timezone":
        task["finished_at"] = "2026-10-24T00:10:00"
    elif change == "backwards-finish":
        task["finished_at"] = START
    else:
        del task["unconfirmed"][0]["resume"]
    with pytest.raises(ledger.LedgerError, match=message):
        rooted.record_task(task, at=AT)


@pytest.mark.parametrize("used", ["dynamic-import", "config-cli", "registry"])
def test_statically_unreferenced_but_used_code_is_kept(rooted: Any, used: str) -> None:
    paths = _paths(used=used)
    finding = _finding(
        kind="unused-code", details={"decision": "unverified", "usage_paths": paths}
    )
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    finding["details"]["decision"] = "delete-candidate"
    with pytest.raises(ledger.LedgerError, match="all usage paths checked"):
        rooted.record_finding(finding, at=AT)
    finding["details"]["decision"] = "keep"
    rooted.record_finding(finding, at=AT)
    restored = ledger.Ledger.load(rooted.path, slot=SLOT, base_sha=BASE)
    assert restored.snapshot["findings"]["f-root"]["details"]["decision"] == "keep"


def test_obsolete_code_can_be_confirmed_after_every_entrypoint_check(
    rooted: Any,
) -> None:
    paths = _paths()
    paths[0].update(
        outcome="not-applicable",
        reason="no generated artifacts are used by this component",
    )
    finding = _finding(
        kind="unused-code",
        details={"decision": "delete-candidate", "usage_paths": paths},
    )
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    rooted.record_finding(finding, at=AT)
    assert rooted.snapshot["findings"]["f-root"]["status"] == "confirmed"


@pytest.mark.parametrize("missing", sorted(ledger.USAGE_PATHS))
def test_deletion_cannot_skip_an_entrypoint(rooted: Any, missing: str) -> None:
    finding = _finding(
        kind="unused-code",
        details={
            "decision": "delete-candidate",
            "usage_paths": [p for p in _paths() if p["kind"] != missing],
        },
    )
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    with pytest.raises(ledger.LedgerError, match="all usage paths checked"):
        rooted.record_finding(finding, at=AT)


@pytest.mark.parametrize(
    "defect",
    ["no-evidence", "unverified-path", "unverified-decision", "no-used-path-for-keep"],
)
def test_unused_code_uncertainty_cannot_be_promoted(rooted: Any, defect: str) -> None:
    finding = _finding(
        kind="unused-code",
        details={"decision": "delete-candidate", "usage_paths": _paths()},
    )
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    if defect == "no-evidence":
        finding["evidence"] = []
    elif defect == "unverified-path":
        finding["details"]["usage_paths"][0].update(
            outcome="unverified", evidence=[], reason="tool unavailable"
        )
    elif defect == "unverified-decision":
        finding["details"]["decision"] = "unverified"
    else:
        finding["details"]["decision"] = "keep"
    with pytest.raises(ledger.LedgerError):
        rooted.record_finding(finding, at=AT)
    restored = ledger.Ledger.load(rooted.path, slot=SLOT, base_sha=BASE)
    assert restored.snapshot["findings"]["f-root"]["status"] == "suspected"


def test_unverified_route_and_refutation_retain_their_evidence(rooted: Any) -> None:
    finding = _finding(
        kind="unused-code",
        details={
            "decision": "unverified",
            "usage_paths": [
                {
                    "kind": "dynamic-import",
                    "outcome": "unverified",
                    "evidence": [],
                    "reason": "package cannot be imported in this environment",
                }
            ],
        },
    )
    finding["status"] = "unverified"
    rooted.record_finding(finding, at=AT)
    finding.update(
        status="refuted",
        counterevidence=["e-code"],
        conclusion="dynamic plugin loader invokes this module",
    )
    finding["details"] = {
        "decision": "keep",
        "usage_paths": _paths(used="dynamic-import"),
    }
    rooted.record_finding(finding, at=AT)
    assert rooted.snapshot["findings"]["f-root"]["counterevidence"] == ["e-code"]


@pytest.mark.parametrize(
    "missing",
    [
        "target_version",
        "conditions",
        "reachability",
        "impact",
        "remediation",
        "verification",
        "usage_paths",
        "evidence",
    ],
)
def test_vulnerability_cannot_be_confirmed_from_a_warning_alone(
    rooted: Any, missing: str
) -> None:
    details = _vulnerability()
    details["usage_paths"] = [
        p for p in _paths(used="config-cli") if p["kind"] == "config-cli"
    ]
    finding = _finding(kind="vulnerability", details=details)
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    if missing == "evidence":
        finding["evidence"] = []
    else:
        details[missing] = [] if missing in {"conditions", "usage_paths"} else ""
    with pytest.raises(ledger.LedgerError):
        rooted.record_finding(finding, at=AT)


def test_vulnerability_target_version_must_match_reachable_evidence(
    rooted: Any,
) -> None:
    details = _vulnerability()
    details["usage_paths"] = [
        p for p in _paths(used="config-cli") if p["kind"] == "config-cli"
    ]
    details["target_version"] = "library==1.2.3"
    finding = _finding(kind="vulnerability", details=details)
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    with pytest.raises(ledger.LedgerError, match="target_version.*matching evidence"):
        rooted.record_finding(finding, at=AT)
    rooted.record_evidence(
        _evidence(
            "e-repro",
            source_kind="isolated-test",
            reference="artifacts/reproduction.json",
            target_version="library==1.2.3",
        ),
        at=AT,
    )
    details["usage_paths"][0]["evidence"] = ["e-repro"]
    rooted.record_finding(finding, at=AT)
    finding.update(
        status="refuted",
        counterevidence=["e-repro"],
        conclusion="caller rejects untrusted input before the dependency",
    )
    rooted.record_finding(finding, at=AT)
    assert rooted.snapshot["findings"]["f-root"]["status"] == "refuted"


@pytest.mark.parametrize(
    "target,message",
    [
        ("finding-task", "unknown task"),
        ("finding-evidence", "unknown evidence"),
        ("counterevidence", "unknown evidence"),
        ("usage-evidence", "unknown evidence"),
        ("handoff", "unknown finding"),
    ],
)
def test_all_record_references_must_exist(
    rooted: Any, target: str, message: str
) -> None:
    finding = _finding()
    if target == "handoff":
        with pytest.raises(ledger.LedgerError, match=message):
            rooted.record_handoff(
                {
                    "id": "h-root",
                    "finding_id": "missing",
                    "kind": "unassigned",
                    "reference": None,
                    "related_parents": [],
                    "reason": "no owner",
                },
                at=AT,
            )
        return
    if target == "finding-task":
        finding["task_id"] = "missing"
    elif target == "finding-evidence":
        finding["evidence"] = ["missing"]
    elif target == "counterevidence":
        finding["counterevidence"] = ["missing"]
    else:
        finding.update(
            kind="unused-code",
            details={"decision": "unverified", "usage_paths": _paths()},
        )
        finding["details"]["usage_paths"][0]["evidence"] = ["missing"]
    with pytest.raises(ledger.LedgerError, match=message):
        rooted.record_finding(finding, at=AT)


@pytest.mark.parametrize(
    "corruption,message",
    [
        ("truncated", "replay"),
        ("duplicate-key", "duplicate JSON key"),
        ("duplicate-nested-key", "duplicate JSON key"),
        ("unknown-version", "unknown ledger version"),
        ("boolean-version", "unknown ledger version"),
        ("empty-history", "non-empty event history"),
        ("bad-sequence", "non-contiguous sequence"),
        ("wrong-slot", "slot/base_sha mismatch"),
        ("wrong-base", "slot/base_sha mismatch"),
        ("unknown-event", "unknown event kind"),
        ("nan", "invalid JSON constant"),
        ("invalid-utf8", "replay"),
        ("unknown-origin", "unknown task"),
        ("illegal-transition", "initial status"),
    ],
)
def test_corrupt_history_stops_with_recovery_instructions(
    rooted: Any, corruption: str, message: str
) -> None:
    path = rooted.path
    doc = json.loads(path.read_text())
    if corruption == "unknown-version":
        doc["version"] = 999
    elif corruption == "boolean-version":
        doc["version"] = True
    elif corruption == "empty-history":
        doc["events"] = []
    elif corruption == "bad-sequence":
        doc["events"][-1]["sequence"] = 99
    elif corruption == "wrong-slot":
        doc["events"][-1]["slot"] = "20261025"
    elif corruption == "wrong-base":
        doc["events"][-1]["base_sha"] = "b" * 40
    elif corruption == "unknown-event":
        doc["events"][-1]["kind"] = "future.event"
    elif corruption == "unknown-origin":
        doc["events"][-1]["data"]["origin"] = {"kind": "task", "id": "missing"}
    elif corruption == "illegal-transition":
        doc["events"][-1]["data"]["status"] = "running"
    raw = json.dumps(doc).encode()
    if corruption == "truncated":
        raw = raw[:-9]
    elif corruption == "duplicate-key":
        raw = raw.replace(b'"version": 1', b'"version": 1, "version": 1')
    elif corruption == "duplicate-nested-key":
        raw = raw.replace(b'"goal":', b'"goal": "overwritten", "goal":')
    elif corruption == "nan":
        raw = raw.replace(b'"version": 1', b'"version": NaN')
    elif corruption == "invalid-utf8":
        raw += b"\xff"
    path.write_bytes(raw)
    with pytest.raises(ledger.LedgerCorruptError, match=message) as exc:
        ledger.Ledger.load(path, slot=SLOT, base_sha=BASE)
    assert "verified backup" in str(exc.value)
    assert "do not reinitialize or discard events" in str(exc.value)
    assert path.read_bytes() == raw
    with pytest.raises(ledger.LedgerError, match="already exists"):
        ledger.Ledger.create(path, slot=SLOT, base_sha=BASE, goal="replacement", at=AT)
    assert path.read_bytes() == raw


def test_missing_ledger_is_not_initialized_implicitly(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(ledger.LedgerCorruptError, match="restore or repair explicitly"):
        ledger.Ledger.load(path, slot=SLOT, base_sha=BASE)
    assert not path.exists()


def test_atomic_save_failure_keeps_the_previous_history(
    book: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, snapshot = book.path.read_bytes(), book.snapshot
    original = ledger.Path.replace

    def fail_replace(path: Path, target: Path) -> Any:
        if target == book.path:
            raise OSError("simulated interrupted replace")
        return original(path, target)

    monkeypatch.setattr(ledger.Path, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted replace"):
        book.record_task(_task("t-new"), at=AT)
    assert book.path.read_bytes() == before
    assert book.snapshot == snapshot
    assert sorted(p.name for p in book.path.parent.iterdir()) == [
        "audit.json",
        "audit.json.lock",
    ]
    monkeypatch.setattr(ledger.Path, "replace", original)
    book.record_task(_task("t-new"), at=AT)
    assert "t-new" in book.snapshot["tasks"]


def test_concurrent_stale_writers_do_not_lose_committed_events(book: Any) -> None:
    writers = [
        ledger.Ledger.load(book.path, slot=SLOT, base_sha=BASE) for _ in range(2)
    ]
    barrier = threading.Barrier(2)

    def write(i: int) -> str:
        barrier.wait(timeout=10)
        try:
            writers[i].record_task(_task(f"t-{i}"), at=AT)
        except ledger.LedgerConflictError as exc:
            assert "reload and reconsider" in str(exc)
            return "stale"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(write, range(2)))
    assert sorted(statuses) == ["committed", "stale"]
    final = ledger.Ledger.load(book.path, slot=SLOT, base_sha=BASE)
    assert len(final.snapshot["tasks"]) == 1
    loser = statuses.index("stale")
    final.record_task(_task(f"t-{loser}"), at=AT)
    assert (
        len(ledger.Ledger.load(book.path, slot=SLOT, base_sha=BASE).snapshot["tasks"])
        == 2
    )


def test_acquisition_versions_and_historical_references_are_immutable(
    book: Any,
) -> None:
    first = _evidence(
        "e-pr-old",
        source_kind="github-pr",
        reference="https://github.com/example/repo/pull/1#diff",
        target_version=None,
        pr_head="b" * 40,
    )
    book.record_evidence(first, at=AT)
    old_ref = book.reference("evidence", "e-pr-old")
    first["pr_head"] = "c" * 40
    with pytest.raises(ledger.LedgerError, match="immutable ID"):
        book.record_evidence(first, at=AT)
    first["id"] = "e-pr-new"
    book.record_evidence(first, at=AT)
    assert book.original(old_ref)["pr_head"] == "b" * 40
    assert book.snapshot["evidence"]["e-pr-new"]["pr_head"] == "c" * 40
    bad_ref = {**old_ref, "sha256": "0" * 64}
    with pytest.raises(ledger.LedgerError, match="content/entity mismatch"):
        book.original(bad_ref)
    with pytest.raises(ledger.LedgerError, match="wrong ledger"):
        book.original({**old_ref, "path": str(book.path.with_name("another.json"))})


def test_summary_pages_decisions_and_resolves_full_originals(book: Any) -> None:
    for i in range(8):
        task = _task(f"t-{i}")
        task["next_questions"] = [
            {"question": "Follow the shared hypothesis", "scope": ["src/shared"]}
        ]
        book.record_task(task, at=AT)
    finding = _finding(task_id="t-0")
    finding["conclusion"] = "hypothesis " * 3000
    book.record_finding(finding, at=AT)
    page = book.summary(limit=3)
    assert len(page["unfinished_tasks"]) == 3
    assert page["totals"]["unfinished_tasks"] == 8
    assert page["remaining"]["unfinished_tasks"] == 5
    assert page["totals"]["next_candidates"] == 9
    second = book.summary(limit=3, offset=3)
    assert [r["id"] for r in second["unfinished_tasks"]] == ["t-3", "t-4", "t-5"]
    row = page["unverified_hypotheses"][0]
    assert len(row["conclusion_excerpt"]) == 240
    assert book.original(row["ref"])["conclusion"] == finding["conclusion"]
    page["unfinished_tasks"][0]["id"] = "mutated"
    snapshot = book.snapshot
    snapshot["tasks"]["t-0"]["origin"]["id"] = "mutated"
    assert book.snapshot["tasks"]["t-0"]["origin"]["id"] == SLOT


def test_run_cannot_claim_success_with_outstanding_work_or_risks(rooted: Any) -> None:
    with pytest.raises(ledger.LedgerError, match="require incomplete"):
        rooted.end(_ending("completed"), at=AT)
    _start(rooted, "t-root")
    _finish(rooted, "t-root")
    rooted.record_finding(_finding(), at=AT)
    with pytest.raises(ledger.LedgerError, match="require incomplete"):
        rooted.end(_ending("completed"), at=AT)
    finding = rooted.snapshot["findings"]["f-root"]
    finding.update(status="refuted", counterevidence=["e-code"])
    rooted.record_finding(finding, at=AT)
    end = _ending("completed")
    end["unverified_risks"] = [GAP.copy()]
    with pytest.raises(ledger.LedgerError, match="require incomplete"):
        rooted.end(end, at=AT)
    rooted.end(_ending("completed"), at=AT)
    with pytest.raises(ledger.LedgerError, match="only an incomplete run"):
        rooted.resume("do more", at=AT)
    with pytest.raises(ledger.LedgerError, match="run is closed"):
        rooted.record_evidence(_evidence("e-after-close"), at=AT)
    rooted.bind_report(1500, at=AT)
    with pytest.raises(ledger.LedgerError, match="already bound"):
        rooted.bind_report(1501, at=AT)


def test_replay_detects_cycles_in_stored_history(rooted: Any) -> None:
    rooted.record_finding(_finding(), at=AT)
    doc = json.loads(rooted.path.read_text())
    event = copy.deepcopy(doc["events"][-2])
    event["sequence"] = len(doc["events"]) + 1
    event["data"]["origin"] = {"kind": "finding", "id": "f-root"}
    doc["events"].append(event)
    rooted.path.write_text(json.dumps(doc))
    with pytest.raises(ledger.LedgerCorruptError, match="origin cycle"):
        ledger.Ledger.load(rooted.path, slot=SLOT, base_sha=BASE)


def test_summary_deduplicates_registered_followups_and_names_retries(
    rooted: Any,
) -> None:
    task = rooted.snapshot["tasks"]["t-root"]
    task["next_questions"] = [
        {"question": task["question"], "scope": task["scope"]},
        {"question": "What invokes the plugin?", "scope": ["configs/plugin.yaml"]},
    ]
    rooted.record_task(task, at=AT)
    rooted.record_task(
        _task(
            "t-plugin",
            question="What invokes the plugin?",
            scope=["configs/plugin.yaml"],
        ),
        at=AT,
    )
    _start(rooted, "t-root")
    _finish(rooted, "t-root", "failed")
    summary = rooted.summary()
    assert summary["totals"]["next_candidates"] == 2
    candidates = {c["source_task_id"]: c for c in summary["next_candidates"]}
    assert candidates["t-root"]["kind"] == "resume-task"
    assert candidates["t-plugin"]["kind"] == "planned-task"
    assert rooted.original(candidates["t-root"]["ref"])["unconfirmed"] == [GAP]


def test_cancelled_scope_must_be_explicitly_excluded_before_success(
    rooted: Any,
) -> None:
    _finish(rooted, "t-root", "cancelled")
    with pytest.raises(ledger.LedgerError, match="cancelled task scope.*excluded"):
        rooted.end(_ending("completed"), at=AT)
    ending = _ending("completed")
    ending["excluded_scope"].append("src/t-root")
    rooted.end(ending, at=AT)
    assert rooted.snapshot["run"]["status"] == "completed"


def test_reference_detects_a_changed_history_prefix(book: Any) -> None:
    book.bind_report(1500, at=AT)
    original_ref = book.reference("run", SLOT)
    doc = json.loads(book.path.read_text())
    doc["events"][0]["data"]["goal"] = "different objective at the same path"
    book.path.write_text(json.dumps(doc))
    restored = ledger.Ledger.load(book.path, slot=SLOT, base_sha=BASE)
    with pytest.raises(ledger.LedgerError, match="content/entity mismatch"):
        restored.original(original_ref)


@pytest.mark.parametrize(
    "defect,message",
    [
        ("missing-version", "target_version or PR head required"),
        ("short-head", "full lowercase commit SHA"),
        ("future-acquisition", "acquired_at is after"),
        ("no-timezone", "timezone required"),
    ],
)
def test_evidence_needs_an_observed_version_and_acquisition_time(
    book: Any, defect: str, message: str
) -> None:
    evidence = _evidence(
        "e-invalid", source_kind="github-pr", target_version=None, pr_head="b" * 40
    )
    if defect == "missing-version":
        evidence["pr_head"] = None
    elif defect == "short-head":
        evidence["pr_head"] = "123abcd"
    elif defect == "future-acquisition":
        evidence["acquired_at"] = "2026-10-25T00:00:00+00:00"
    else:
        evidence["acquired_at"] = "2026-10-24T00:00:00"
    with pytest.raises(ledger.LedgerError, match=message):
        book.record_evidence(evidence, at=AT)


def test_native_identity_and_question_cannot_be_rewritten_mid_attempt(
    rooted: Any,
) -> None:
    _start(rooted, "t-root")
    task = rooted.snapshot["tasks"]["t-root"]
    task["agent_id"] = "a-different-native-agent"
    with pytest.raises(ledger.LedgerError, match="changed within an attempt"):
        rooted.record_task(task, at=AT)
    task = rooted.snapshot["tasks"]["t-root"]
    task["question"] = (
        "replace the question while pretending this is the same investigation"
    )
    with pytest.raises(ledger.LedgerError, match="question/scope are immutable"):
        rooted.record_task(task, at=AT)
    second = _task("t-second")
    rooted.record_task(second, at=AT)
    second.update(status="running", agent_id="native-t-root", started_at=AT)
    with pytest.raises(ledger.LedgerError, match="already owns a running task"):
        rooted.record_task(second, at=AT)


@pytest.mark.parametrize("status", ["confirmed", "refuted"])
def test_resolved_findings_cannot_skip_initial_hypotheses(
    rooted: Any, status: str
) -> None:
    finding = _finding()
    finding.update(status=status, evidence=["e-code"], counterevidence=["e-code"])
    with pytest.raises(ledger.LedgerError, match="initial status"):
        rooted.record_finding(finding, at=AT)


def test_refuted_finding_requires_a_reopened_hypothesis_before_confirmation(
    rooted: Any,
) -> None:
    finding = _finding()
    rooted.record_finding(finding, at=AT)
    finding.update(status="refuted", counterevidence=["e-code"])
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    with pytest.raises(
        ledger.LedgerError, match="invalid transition refuted -> confirmed"
    ):
        rooted.record_finding(finding, at=AT)
    finding["status"] = "unverified"
    rooted.record_finding(finding, at=AT)
    finding["status"] = "confirmed"
    rooted.record_finding(finding, at=AT)


def test_failure_to_sync_the_temporary_file_never_replaces_history(
    book: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = book.path.read_bytes()

    def fail_sync(fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(ledger.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="fsync failure"):
        book.record_task(_task("t-new"), at=AT)
    assert book.path.read_bytes() == before
    assert "t-new" not in book.snapshot["tasks"]


def test_bug_conditions_are_separate_from_the_hypothesis(rooted: Any) -> None:
    details = _vulnerability()
    finding = _finding(kind="bug", details=details)
    rooted.record_finding(finding, at=AT)
    finding.update(status="confirmed", evidence=["e-code"])
    details["conditions"] = []
    with pytest.raises(ledger.LedgerError, match="bug.conditions"):
        rooted.record_finding(finding, at=AT)
    details["conditions"] = ["a configured input reaches the broken branch"]
    rooted.record_finding(finding, at=AT)
    assert rooted.original(rooted.reference("finding", "f-root"))["details"] == details
