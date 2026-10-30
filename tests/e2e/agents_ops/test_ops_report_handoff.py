"""Ledger rendering and reobservation use the unchanged proposal/triage contract."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path.insert(0, str(OPS / "ops-report"))
sys.path.insert(0, str(OPS))
ledger: Any = importlib.import_module("opsreport.ledger")
handoff: Any = importlib.import_module("opsreport.handoff")
report: Any = importlib.import_module("opsreport.report")
proposals: Any = importlib.import_module("opsreport.proposals")
formats: Any = importlib.import_module("issues")

BASE = "a" * 40
SLOT = "20261024"
AT = "2026-10-24T01:00:00+00:00"
GAP = {
    "scope": "src/hidden",
    "reason": "tool failed",
    "resume": "restore tool then retry",
}


def _book(path: Path, *, slot: str = SLOT, base: str = BASE) -> Any:
    book = ledger.Ledger.create(
        path,
        slot=slot,
        base_sha=base,
        goal="Verify potential problems",
        at=AT,
    )
    book.record_evidence(
        {
            "id": "e-code",
            "source_kind": "code",
            "reference": "src/check.py:20",
            "acquired_at": AT,
            "target_version": base,
            "pr_head": None,
        },
        at=AT,
    )
    _task(book, "t-root")
    return book


def _task(
    book: Any,
    key: str,
    *,
    origin: dict[str, str] | None = None,
    question: str | None = None,
) -> None:
    row = {
        "id": key,
        "question": question or f"Verify {key}",
        "scope": [f"src/{key}"],
        "origin": origin or {"kind": "goal", "id": book.snapshot["slot"]},
        "agent_id": None,
        "status": "planned",
        "started_at": None,
        "finished_at": None,
        "conclusion": "",
        "evidence": [],
        "next_questions": [],
        "unconfirmed": [],
    }
    book.record_task(row, at=AT)
    row.update(status="running", agent_id=f"native-{key}", started_at=AT)
    book.record_task(row, at=AT)
    row.update(
        status="completed",
        finished_at=AT,
        conclusion="Execution paths checked",
        evidence=["e-code"],
    )
    book.record_task(row, at=AT)


def _finding(
    book: Any,
    key: str,
    status: str = "confirmed",
    *,
    kind: str = "design",
    details: dict[str, Any] | None = None,
    conclusion: str | None = None,
    task: str = "t-root",
) -> dict[str, Any]:
    row = {
        "id": key,
        "task_id": task,
        "kind": kind,
        "status": "suspected",
        "conclusion": conclusion or f"Conclusion for {key}",
        "evidence": ["e-code"],
        "counterevidence": [],
        "details": details or {},
    }
    book.record_finding(row, at=AT)
    if status != "suspected":
        row.update(
            status=status, counterevidence=["e-code"] if status == "refuted" else []
        )
        book.record_finding(row, at=AT)
    return row


def _route(
    book: Any,
    finding: str,
    kind: str,
    reference: str | None,
    *,
    parents: list[int] | None = None,
    key: str | None = None,
) -> None:
    book.record_handoff(
        {
            "id": key or f"h-{finding}-{kind}",
            "finding_id": finding,
            "kind": kind,
            "reference": reference,
            "related_parents": parents or [],
            "reason": "Checked existing work and its agreed scope",
        },
        at=AT,
    )


def _end(book: Any, *, complete: bool = True) -> None:
    handoff.record_proposal_handoffs(book, at=AT)
    book.end(
        {
            "status": "completed" if complete else "incomplete",
            "reason": "Important hypotheses verified"
            if complete
            else "Important checks remain",
            "checked_scope": ["src/check.py", "CLI and registry"],
            "excluded_scope": ["external submodule internals"],
            "unverified_risks": [] if complete else [GAP],
            "further_investigation": [] if complete else ["restore the missing tool"],
        },
        at=AT,
    )


def _render(book: Any, **changes: Any) -> tuple[str, str, list[str]]:
    result: tuple[str, str, list[str]] = report.render_ledger_report(
        **{
            "ledger": book,
            "agent_name": "codex",
            "since": datetime.fromisoformat("2026-10-21T01:00:00+00:00"),
            "until": datetime.fromisoformat(AT),
            "input_sha256": "b" * 64,
            "ledger_reference": "https://example.test/archive/ledger.json",
            **changes,
        }
    )
    return result


def _paths(used: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "outcome": "used" if used == kind else "unused",
            "evidence": ["e-code"],
            "reason": "pinned entrypoint verified",
        }
        for kind in sorted(ledger.USAGE_PATHS)
    ]


def _bug_details() -> dict[str, Any]:
    return {
        "target_version": BASE,
        "conditions": ["untrusted CLI path"],
        "reachability": "CLI forwards input to the unsafe call",
        "impact": "arbitrary file read",
        "remediation": "validate the path",
        "verification": "isolated traversal regression fixture",
        "usage_paths": [p for p in _paths("config-cli") if p["outcome"] == "used"],
    }


def test_only_confirmed_findings_are_proposals_and_all_states_are_visible(
    tmp_path: Path,
) -> None:
    book = _book(tmp_path / "ledger.json")
    for status in ("confirmed", "suspected", "unverified", "refuted"):
        _finding(book, f"f-{status}", status)
    _task(book, "t-followup", origin={"kind": "finding", "id": "f-confirmed"})
    _task(book, "t-impact", origin={"kind": "task", "id": "t-followup"})
    _end(book, complete=False)
    title, body, ids = _render(book)
    meta, parsed = proposals.parse_report(body)
    assert title == "運用レポート 2026-10-24"
    assert ids == ["R-20261024-01"]
    assert meta.status == "incomplete" and meta.base == BASE and meta.audit == "b" * 64
    assert "**未完了**" in body
    assert "f-confirmed" in parsed[0].field("根拠")
    assert all(
        f"f-{s}" not in parsed[0].field("根拠")
        for s in ("suspected", "unverified", "refuted")
    )
    for status in ("confirmed", "suspected", "unverified", "refuted"):
        assert f"### {status}（1件）" in body
        assert f"f-{status}" in body
    assert "反証: e-code" in body
    assert "Important checks remain" in body and "external submodule internals" in body
    assert "外部submodule内部は対象外・未調査" in body
    assert "src/hidden / 理由: tool failed / 再開: restore tool then retry" in body
    assert "タスク: 3件 / completed 3件" in body
    assert (
        "goal:20261024 → task:t-root → finding:f-confirmed → task:t-followup → task:t-impact"
        in body
    )
    assert str(book.path) not in body


def test_existing_targets_parent_new_and_unassigned_routes_are_distinct(
    tmp_path: Path,
) -> None:
    book = _book(tmp_path / "ledger.json")
    for key in ("existing-issue", "existing-pr", "parent", "new", "unassigned"):
        _finding(book, key)
    _route(book, "existing-issue", "issue", "#77")
    _route(book, "existing-pr", "pr", "https://github.com/o/r/pull/88")
    _route(book, "existing-pr", "proposal", "pending", parents=[1])
    _route(book, "parent", "proposal", "pending", parents=[1])
    _route(book, "new", "proposal", "new, unagreed direction")
    _route(book, "unassigned", "unassigned", None)
    _end(book)
    _, body, ids = _render(book)
    meta, parsed = proposals.parse_report(body)
    assert meta.status == "complete" and len(ids) == 3
    assert "発見 existing-issue → issue #77" in body
    assert "発見 existing-pr → pr #88" in body
    assert {p.parent() for p in parsed} == {None, 1}
    owned = next(p for p in parsed if p.parent() == 1)
    assert "発見 parent" in owned.field("根拠")
    assert "関連親 #1（範囲確認・受領待ち）" in body
    assert "新規・未合意 / 未割当（担当親未指定）" in body
    for prop in parsed:
        prop.requirements()
        assert prop.selection() == (False, None)
        assert "existing-issue" not in prop.field(
            "根拠"
        ) and "existing-pr" not in prop.field("根拠")
    assert book.snapshot["handoffs"]["h-parent-proposal"]["reference"] == owned.pid


@pytest.mark.parametrize("used", ["dynamic-import", "registry", "config-cli"])
def test_unused_candidates_show_verified_usage_and_keep_is_not_a_fix(
    tmp_path: Path, used: str
) -> None:
    book = _book(tmp_path / "ledger.json")
    _finding(
        book,
        "delete",
        kind="unused-code",
        details={"decision": "delete-candidate", "usage_paths": _paths()},
    )
    _finding(
        book,
        "keep",
        kind="unused-code",
        details={"decision": "keep", "usage_paths": _paths(used)},
    )
    _finding(
        book,
        "unknown",
        "unverified",
        kind="unused-code",
        details={"decision": "delete-candidate", "usage_paths": []},
    )
    _end(book, complete=False)
    _, body, ids = _render(book)
    assert ids == ["R-20261024-01"]
    assert "発見 delete" in proposals.parse_report(body)[1][0].field("根拠")
    for label in ("削除候補", "維持", "未確認"):
        assert f"### {label}（1件）" in body
    assert f"利用経路 {used}: used / pinned entrypoint verified / 根拠: e-code" in body
    assert "利用経路: 未確認（確認記録なし）" in body


def test_vulnerability_requires_reachable_versioned_evidence(tmp_path: Path) -> None:
    book = _book(tmp_path / "ledger.json")
    _finding(book, "reachable", kind="vulnerability", details=_bug_details())
    warning = _bug_details()
    warning.update(reachability="", usage_paths=[])
    row = _finding(
        book, "dependency-warning", "suspected", kind="vulnerability", details=warning
    )
    row["status"] = "confirmed"
    with pytest.raises(ledger.LedgerError, match="reachability|reachable"):
        book.record_finding(row, at=AT)
    _end(book, complete=False)
    _, body, ids = _render(book)
    assert len(ids) == 1
    assert "到達可能性: CLI forwards input to the unsafe call" in body
    assert "発生条件: untrusted CLI path" in body
    assert "到達可能性: 未確認" in body
    assert "src/check.py:20" in body and "取得 " + AT in body
    prop = proposals.parse_report(body)[1][0]
    assert "発見 reachable" in prop.field("根拠")
    assert prop.field("受入条件") == "isolated traversal regression fixture"


def test_preparing_records_is_idempotent_and_required_before_end(
    tmp_path: Path,
) -> None:
    book = _book(tmp_path / "ledger.json")
    _finding(book, "first")
    with pytest.raises(ledger.LedgerError, match="end the ledger"):
        _render(book)
    with pytest.raises(ledger.LedgerError, match="mapping missing"):
        handoff.finding_proposals(book)
    handoff.record_proposal_handoffs(book, at=AT)
    before = book.path.read_bytes()
    handoff.record_proposal_handoffs(book, at=AT)
    assert book.path.read_bytes() == before
    _end(book)
    with pytest.raises(ledger.LedgerError, match="before end"):
        handoff.record_proposal_handoffs(book, at=AT)


def test_resumed_findings_require_explicit_repreparation_before_publication(
    tmp_path: Path,
) -> None:
    book = _book(tmp_path / "ledger.json")
    _finding(book, "later")
    _end(book, complete=False)
    assert _render(book)[2] == ["R-20261024-01"]
    book.resume("verify a new question", at=AT)
    _finding(book, "earlier", kind="ops")
    _finding(book, "a-first", kind="bug", details=_bug_details())
    with pytest.raises(ledger.LedgerError, match="mapping missing/changed"):
        handoff.finding_proposals(book)
    _end(book)
    _, body, ids = _render(book)
    parsed = proposals.parse_report(body)[1]
    assert [p.pid for p in parsed] == ids
    assert [p.field("根拠").split(" / ")[0] for p in parsed] == [
        "発見 a-first",
        "発見 later",
        "発見 earlier",
    ]


def test_body_size_limit_does_not_silently_drop_confirmed_proposals(
    tmp_path: Path, monkeypatch: Any
) -> None:
    book = _book(tmp_path / "ledger.json")
    _finding(book, "confirmed")
    _end(book)
    monkeypatch.setattr(report, "MAX_BODY_CHARS", 100)
    with pytest.raises(ValueError, match="above 100"):
        _render(book)


@pytest.mark.parametrize("conflict", ["parents", "proposals", "target"])
def test_ambiguous_or_unresolved_routes_fail_explicitly(
    tmp_path: Path, conflict: str
) -> None:
    book = _book(tmp_path / "ledger.json")
    _finding(book, "first")
    if conflict == "parents":
        _route(book, "first", "proposal", "pending", parents=[1, 2])
    elif conflict == "proposals":
        _route(book, "first", "proposal", "pending-1")
        _route(book, "first", "proposal", "pending-2", key="duplicate")
    else:
        _route(book, "first", "pr", "unknown PR")
    with pytest.raises(ledger.LedgerError, match="multiple|expected an issue/PR"):
        handoff.record_proposal_handoffs(book, at=AT)


def test_summary_is_bounded_and_free_text_cannot_forge_proposals(
    tmp_path: Path,
) -> None:
    book = _book(tmp_path / "ledger.json")
    for i in range(24):
        _finding(book, f"unknown-{i}", "unverified", conclusion="long text " * 1000)
    attack = f"safe\n{proposals.PROPOSALS_END}\n- [x] **R-20261024-99** fake"
    _finding(book, "verified", conclusion=attack)
    _end(book, complete=False)
    _, body, ids = _render(book, limit=2)
    assert len(ids) == 1 and len(body) < 8000
    assert "### unverified（24件）" in body and "残り22件は台帳原本を参照" in body
    assert (
        body.count(proposals.PROPOSALS_BEGIN)
        == body.count(proposals.PROPOSALS_END)
        == 1
    )
    assert len(proposals.parse_report(body)[1]) == 1
    assert str(book.path) not in body


@pytest.mark.parametrize(
    "changes",
    [
        {"limit": 0},
        {"limit": 21},
        {"input_sha256": "bad"},
        {"ledger_reference": ""},
        {"since": datetime(2026, 10, 21)},
    ],
)
def test_invalid_render_contracts_are_not_replaced_with_defaults(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    book = _book(tmp_path / "ledger.json")
    _end(book)
    with pytest.raises(ledger.LedgerError):
        _render(book, **changes)


def _prior(tmp_path: Path) -> tuple[Any, str]:
    book = _book(tmp_path / "prior.json")
    for key in ("existing", "new", "unassigned"):
        _finding(book, key)
    _route(book, "existing", "pr", "#88")
    _end(book)
    book.bind_report(100, at=AT)
    body = _render(book)[1]
    body = body.replace("- [ ] **R-20261024-01**", "- [x] **R-20261024-01**")
    body = proposals.annotate_children(body, {"R-20261024-01": 101})
    return book, body


def test_two_rounds_recover_issue_pr_and_finding_links_for_new_base(
    tmp_path: Path,
) -> None:
    previous, body = _prior(tmp_path)
    previous = ledger.Ledger.load(previous.path, slot=SLOT, base_sha=BASE)
    context = handoff.reobservation_input(previous, report_issue=100, report_body=body)
    assert context["total"] == 3 and context["remaining"] == 0
    assert context["source_base_sha"] == BASE and context["source_slot"] == SLOT
    assert {(r["finding_id"], r["kind"], r["number"]) for r in context["items"]} == {
        ("existing", "pr", 88),
        ("new", "issue", 101),
        ("unassigned", "proposal", None),
    }
    for item in context["items"]:
        assert previous.original(item["finding_ref"])["id"] == item["finding_id"]
        assert (
            previous.original(item["handoff_ref"])["finding_id"] == item["finding_id"]
        )
    second = _book(tmp_path / "next.json", slot="20261027", base="c" * 40)
    input_path = tmp_path / "previous-handoffs.json"
    input_path.write_text(json.dumps(context), encoding="utf-8")
    loaded = json.loads(input_path.read_text())
    for i, item in enumerate(loaded["items"]):
        _task(second, f"reobserve-{i}", question=item["question"])
        assert "発見 " + item["finding_id"] in item["question"]
        assert (
            second.snapshot["tasks"][f"reobserve-{i}"]["question"] == item["question"]
        )
    # A next-round result uses the next base's fresh evidence, not the prior SHA.
    _finding(second, "fixed-new", "refuted", task="reobserve-1")
    _end(second)
    assert _render(second)[2] == []
    assert second.snapshot["base_sha"] != loaded["source_base_sha"]
    assert previous.snapshot["findings"]["new"]["status"] == "confirmed"


def test_reobservation_is_paged_and_keeps_unadopted_proposals(tmp_path: Path) -> None:
    book, body = _prior(tmp_path)
    pages = [
        handoff.reobservation_input(
            book, report_issue=100, report_body=body, limit=1, offset=i
        )
        for i in range(3)
    ]
    assert [p["remaining"] for p in pages] == [2, 1, 0]
    assert [p["items"][0]["finding_id"] for p in pages] == [
        "existing",
        "new",
        "unassigned",
    ]
    assert pages[-1]["items"][0]["adoption"] == "undecided"
    assert "採否・引渡先" in pages[-1]["items"][0]["question"]


@pytest.mark.parametrize(
    "change", ["issue", "slot", "base", "count", "text", "deferral"]
)
def test_reobservation_refuses_unmatched_or_inconsistent_publication(
    tmp_path: Path, change: str
) -> None:
    book, body = _prior(tmp_path)
    report_issue = 100
    if change == "issue":
        report_issue = 999
    elif change == "slot":
        body = body.replace("slot=20261024", "slot=20261027")
    elif change == "base":
        body = body.replace(f"base={BASE}", "base=" + "c" * 40)
    elif change == "count":
        body = body.replace("proposals=2", "proposals=3")
    elif change == "text":
        body = body.replace("Conclusion for new", "different proposed work")
    else:
        body = body.replace("見送り理由: 未記入", "見送り理由: defer", 1)
    with pytest.raises(ValueError):
        handoff.reobservation_input(book, report_issue=report_issue, report_body=body)


def test_existing_child_is_retained_without_inferring_adoption_from_unchecked_box(
    tmp_path: Path,
) -> None:
    book, body = _prior(tmp_path)
    body = body.replace("- [x] **R-20261024-01**", "- [ ] **R-20261024-01**")
    page = handoff.reobservation_input(book, report_issue=100, report_body=body)
    item = next(row for row in page["items"] if row["finding_id"] == "new")
    assert item["kind"] == "issue" and item["number"] == 101
    assert item["adoption"] == "undecided"


def test_incomplete_round_must_resume_instead_of_becoming_next_input(
    tmp_path: Path,
) -> None:
    book = _book(tmp_path / "ledger.json")
    _end(book, complete=False)
    book.bind_report(100, at=AT)
    with pytest.raises(ledger.LedgerError, match="resume incomplete"):
        handoff.reobservation_input(
            book, report_issue=100, report_body=_render(book)[1]
        )


def test_unowned_ledger_proposal_enters_existing_triage_organization(
    ops_env: Any, tmp_path: Path
) -> None:
    ops = ops_env
    generated = ops.run("report")
    assert generated.returncode == 0, generated.stderr
    book = _book(tmp_path / "ledger.json", slot="20261012")
    _finding(book, "new-unagreed")
    _end(book)
    state = ops.state
    row = next(i for i in state["issues"] if i["number"] == 100)
    row["body"] = _render(book)[1].replace(
        "- [ ] **R-20261012-01**", "- [x] **R-20261012-01**"
    )
    ops.write_state(state)
    result = ops.run("triage")
    assert result.returncode == 0, result.stderr
    calls = (ops.tmp / "organization_calls.jsonl").read_text().splitlines()
    assert len(calls) == 1
    inputs = [
        path for run in ops.run_dirs() for path in run.glob("organization-input.json")
    ]
    assert len(inputs) == 1
    fixed = json.loads(inputs[0].read_text())
    assert len(fixed["proposals"]) == 1
    assert "R-20261012-01" in json.dumps(fixed["proposals"])
    assert "new-unagreed" in json.dumps(fixed["proposals"])
    assert (
        "評価対象の選択が不足している"
        in next(i for i in ops.state["issues"] if i["number"] == 100)["body"]
    )
    assert not [i for i in ops.state["issues"] if "issue:child" in i["labels"]]


def test_ledger_report_uses_existing_triage_owner_receipt_without_agent_start(
    ops_env: Any, tmp_path: Path
) -> None:
    ops = ops_env
    # Only the existing fake report setup creates the target issue; CLI generation
    # stays outside this change. Replace it with the new library's body.
    generated = ops.run("report")
    assert generated.returncode == 0, generated.stderr
    state = ops.state
    row = next(i for i in state["issues"] if i["number"] == 100)
    parent_sections = {
        s: "合意済み作業の目的と範囲。"
        for s in formats.PARENT_REQUIRED
        if s != formats.TASKS_HEADING
    }
    state["issues"].append(
        {
            **row,
            "number": 1,
            "title": "[改善] 担当親",
            "body": formats.render_parent_body(parent_sections, []),
            "labels": ["issue:parent"],
            "url": "https://github.com/o/r/issues/1",
        }
    )
    book = _book(tmp_path / "ledger.json", slot="20261012")
    _finding(book, "owned")
    _route(book, "owned", "proposal", "pending", parents=[1])
    _end(book)
    row["body"] = _render(book)[1].replace(
        "- [ ] **R-20261012-01**", "- [x] **R-20261012-01**"
    )
    ops.write_state(state)
    clean_env = {
        k: v
        for k, v in ops.env.items()
        if k not in {"CODEX_THREAD_ID", "CODEX_SESSION_ID", "CLAUDE_CODE_SESSION_ID"}
        and not k.startswith("HERDR_")
    }
    registered = subprocess.run(
        [sys.executable, "-m", "parent", "--repo", "o/r", "--parent", "1", "register"],
        cwd=ops.repo,
        env={
            **clean_env,
            "PYTHONPATH": str(OPS),
            "CLAUDE_CODE_SESSION_ID": "fake-owner",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert registered.returncode == 0, registered.stderr
    result = ops.run("triage")
    assert result.returncode == 0, result.stderr
    updated = next(i for i in ops.state["issues"] if i["number"] == 100)["body"]
    assert "親 #1: 採用済み提案の受領待ち" in updated
    events = [
        json.loads(c["body"].splitlines()[0][len("<!-- parent-session:v1 ") : -4])
        for c in ops.state.get("comments", {}).get("1", [])
    ]
    assert sum(e["event"] == "proposal_pending" for e in events) == 1
    assert not [i for i in ops.state["issues"] if "issue:child" in i["labels"]]
    assert not (ops.tmp / "organization_calls.jsonl").exists()
