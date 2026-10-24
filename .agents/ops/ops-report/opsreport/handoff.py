"""Confirmed ledger findings enter the existing proposal/triage contract.

Routing decisions are ledger records, not permission to implement or start an
owner session. Publication and fetching the next round's inputs belong to the
caller. This module makes no GitHub calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .ledger import Ledger, LedgerError
from .proposals import (
    CATEGORIES,
    PROPOSALS_BEGIN,
    PROPOSALS_END,
    Proposal,
    neutralize,
    parse_report,
    proposal_id,
    render_proposals,
)

_SECTIONS = {
    "bug": "code-debt",
    "vulnerability": "code-debt",
    "unused-code": "code-debt",
    "duplicate": "code-debt",
    "stale-dependency": "code-debt",
    "design": "architecture",
    "ops": "control-rules",
}


def excerpt(value: str, limit: int = 240) -> str:
    text = " ".join(neutralize(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def issue_number(kind: str, reference: str) -> int:
    """Resolve an explicit local number or GitHub issue/PR URL; never guess."""
    match = re.fullmatch(r"#?([1-9][0-9]*)", reference)
    if match:
        return int(match[1])
    resource = {"issue": "issues", "pr": "pull"}[kind]
    match = re.fullmatch(
        rf"https://github\.com/[^/\s]+/[^/\s]+/{resource}/([1-9][0-9]*)/?",
        reference,
    )
    if not match:
        raise LedgerError(f"{kind}: expected an issue/PR number or URL: {reference!r}")
    return int(match[1])


def evidence_text(state: dict[str, Any], refs: list[str], limit: int = 3) -> str:
    parts = []
    for key in refs[:limit]:
        row = state["evidence"][key]
        version = row["pr_head"] or row["target_version"]
        parts.append(
            f"{excerpt(key)}: {excerpt(row['reference'])}"
            f" @ {excerpt(version)} / 取得 {row['acquired_at']}"
        )
    if len(refs) > limit:
        parts.append(f"残り{len(refs) - limit}件は台帳原本参照")
    return " / ".join(parts) or "なし"


def _routes(state: dict[str, Any], finding_id: str) -> list[dict[str, Any]]:
    return [h for h in state["handoffs"].values() if h["finding_id"] == finding_id]


def _proposal_route(routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    proposals = [h for h in routes if h["kind"] == "proposal"]
    if len(proposals) > 1:
        raise LedgerError("finding has multiple proposal routes; reconcile explicitly")
    return proposals[0] if proposals else None


def _reason(routes: list[dict[str, Any]]) -> str:
    proposal = _proposal_route(routes)
    if proposal:
        return str(proposal["reason"])
    return " / ".join(dict.fromkeys(h["reason"] for h in routes)) or (
        "新規・未合意。担当親は未割当。方向性の採用と要件合意を既存triage経路で確認する。"
    )


@dataclass(frozen=True)
class FindingProposal:
    finding_id: str
    pid: str
    proposal: Proposal


def finding_proposals(
    book: Ledger, *, require_recorded: bool = True
) -> list[FindingProposal]:
    """Convert verified problems only, ordered exactly as render_proposals.

    An existing issue/PR always wins over a proposed new issue. A confirmed
    unused-code ``keep`` decision is an observation, not a problem to fix.
    Multiple candidate parents cannot fit the single-parent triage contract.
    """
    state = book.snapshot
    pending: list[tuple[str, Proposal]] = []
    for finding in state["findings"].values():
        if finding["status"] != "confirmed" or (
            finding["kind"] == "unused-code"
            and finding["details"]["decision"] == "keep"
        ):
            continue
        routes = _routes(state, finding["id"])
        existing = [h for h in routes if h["kind"] in {"issue", "pr"}]
        if existing:
            for route in existing:
                issue_number(route["kind"], route["reference"])
            continue
        parents = {p for h in routes for p in h["related_parents"]}
        if len(parents) > 1:
            raise LedgerError(
                f"{finding['id']}: multiple related parents; choose a routing owner explicitly"
            )
        parent = next(iter(parents), None)
        task = state["tasks"][finding["task_id"]]
        details = finding["details"]
        route_label = (
            f"関連親 #{parent}（範囲確認・受領待ち）"
            if parent is not None
            else "新規・未合意 / 未割当（担当親未指定）"
        )
        verification = details.get("verification") or (
            "台帳の根拠・反証と利用経路を再確認し、修正した範囲の回帰検証を記録する。"
        )
        pending.append(
            (
                finding["id"],
                Proposal(
                    section=CATEGORIES[_SECTIONS.get(finding["kind"], "other")],
                    title=excerpt(finding["conclusion"], 160),
                    detail=(
                        f"発見 {finding['id']} / タスク {task['id']} / confirmed\n"
                        f"引渡先: {route_label}\n判断理由: {_reason(routes)}\n"
                        + (
                            f"修正方針: {details['remediation']}"
                            if "remediation" in details
                            else "具体的な変更は担当親が根拠と影響範囲を確認して合意する。"
                        )
                    ),
                    evidence=f"発見 {excerpt(finding['id'])} / base={state['base_sha']} / "
                    + evidence_text(state, finding["evidence"]),
                    purpose=excerpt(finding["conclusion"]),
                    scope=" / ".join(task["scope"]),
                    expected_effect=(
                        f"確認済みの影響を解消する: {details['impact']}"
                        if "impact" in details
                        else "確認済みの問題を解消し、修正結果を次回の調査で再確認できる。"
                    ),
                    acceptance=(verification,),
                    related_parent=parent,
                ),
            )
        )
    sections = list(CATEGORIES.values())
    pending.sort(key=lambda item: (sections.index(item[1].section), item[0]))
    if len(pending) > 999:
        raise LedgerError(
            "more than 999 proposals exceed the existing proposal ID contract"
        )
    result = []
    for index, (key, proposal) in enumerate(pending, 1):
        pid = proposal_id(state["slot"], index)
        if require_recorded:
            proposal_route = _proposal_route(_routes(state, key))
            if proposal_route is None or proposal_route["reference"] != pid:
                raise LedgerError(
                    f"{key}: proposal mapping missing/changed; record_proposal_handoffs before ending the run"
                )
        result.append(FindingProposal(key, pid, proposal))
    return result


def record_proposal_handoffs(book: Ledger, *, at: str) -> list[FindingProposal]:
    """Persist finding → proposal IDs before end(); repeated preparation is inert.

    A caller may first record a proposed route with a descriptive reference and
    related parent/reason. This explicit preparation replaces that reference with
    the final ID. Reprepare after changing findings in a resumed incomplete run.
    """
    state = book.snapshot
    if state["run"]["status"] != "running":
        raise LedgerError(
            "prepare proposal handoffs while the run is running, before end"
        )
    entries = finding_proposals(book, require_recorded=False)
    for entry in entries:
        routes = _routes(state, entry.finding_id)
        old = _proposal_route(routes)
        row = {
            "id": old["id"] if old else f"report-proposal/{entry.finding_id}",
            "finding_id": entry.finding_id,
            "kind": "proposal",
            "reference": entry.pid,
            "related_parents": [entry.proposal.related_parent]
            if entry.proposal.related_parent is not None
            else [],
            "reason": _reason(routes),
        }
        if row != old:
            book.record_handoff(row, at=at)
    return finding_proposals(book)


def reobservation_input(
    book: Ledger,
    *,
    report_issue: int,
    report_body: str,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Page prior handoffs, resolving triage's child annotations for the next run.

    Fetch the bound report's current body even if closed. Verify its immutable
    proposal text; adoption, deferral, parent routing and child/status annotations
    may change through the existing human/triage workflow. No old evidence is
    promoted to proof about the next base SHA.
    """
    if type(limit) is not int or limit < 1 or type(offset) is not int or offset < 0:
        raise LedgerError("positive limit and non-negative offset required")
    state = book.snapshot
    if state["run"]["status"] != "completed":
        raise LedgerError(
            "next-round input requires a completed round; resume incomplete rounds"
        )
    if state["run"]["report_issue"] != report_issue or type(report_issue) is not int:
        raise LedgerError("fetched report issue does not match the bound ledger report")
    meta, parsed = parse_report(report_body)
    if (meta.slot, meta.base, meta.status) != (
        state["slot"],
        state["base_sha"],
        "complete",
    ):
        raise LedgerError("report slot/base/status does not match the source ledger")
    entries = finding_proposals(book)
    markdown, ids = render_proposals([e.proposal for e in entries], state["slot"])
    expected = parse_report(
        "\n".join(
            [report_body.splitlines()[0], PROPOSALS_BEGIN, markdown, PROPOSALS_END]
        )
    )[1]
    if (
        meta.proposal_count != len(ids)
        or [p.pid for p in parsed] != ids
        or [p.snapshot() for p in parsed] != [p.snapshot() for p in expected]
    ):
        raise LedgerError(
            "published proposals do not match the ledger's finding mappings"
        )
    items: list[dict[str, Any]] = []
    seen = set()

    def add(finding_id: str, route: dict[str, Any], **extra: Any) -> None:
        kind, reference = route["kind"], route["reference"]
        number = issue_number(kind, reference) if kind in {"issue", "pr"} else None
        key = (finding_id, kind, number or reference)
        if key in seen:
            return
        seen.add(key)
        finding = state["findings"][finding_id]
        items.append(
            {
                "finding_id": finding_id,
                "finding_status": finding["status"],
                "conclusion_excerpt": excerpt(finding["conclusion"]),
                "finding_ref": book.reference("finding", finding_id),
                "handoff_ref": book.reference("handoff", route["id"]),
                "kind": kind,
                "number": number,
                "reference": reference,
                "related_parents": route["related_parents"],
                "reason_excerpt": excerpt(route["reason"]),
                "question": (
                    f"発見 {excerpt(finding_id)}: {kind} {excerpt(reference)} の現在の状態・"
                    "修正差分と回帰検証を取得し、新しい基準版で修正結果を再確認する。"
                    if number is not None
                    else f"発見 {excerpt(finding_id)}: 前回提案 {reference} の採否・引渡先を確認する。"
                ),
                **extra,
            }
        )

    for route in state["handoffs"].values():
        if route["kind"] in {"issue", "pr"}:
            add(route["finding_id"], route)
    for entry, prop in zip(entries, parsed, strict=True):
        prop.requirements()
        selected, _ = prop.selection()
        deferred = prop.deferred_reason()
        if selected and deferred:
            raise LedgerError(f"{prop.pid}: both adopted and deferred")
        route = _proposal_route(_routes(state, entry.finding_id))
        assert route is not None  # finding_proposals requires the recorded mapping.
        parent = prop.parent()
        add(
            entry.finding_id,
            {
                **route,
                "kind": "issue" if prop.child is not None else "proposal",
                "reference": f"#{prop.child}" if prop.child is not None else prop.pid,
                "related_parents": [parent] if parent is not None else [],
            },
            proposal_id=prop.pid,
            adoption="adopted" if selected else "deferred" if deferred else "undecided",
            deferred_reason=deferred,
        )
    return {
        "source_slot": state["slot"],
        "source_base_sha": state["base_sha"],
        "report_issue": report_issue,
        "run_ref": book.reference("run", state["slot"]),
        "items": items[offset : offset + limit],
        "offset": offset,
        "total": len(items),
        "remaining": max(0, len(items) - offset - limit),
    }
