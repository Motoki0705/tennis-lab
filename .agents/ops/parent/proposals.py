"""Adoption delivery and explicit receipt; neither operation authorizes execution."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues import ProposalSource

from .policy import digest, events, number, object_value

# Reports keep their CLI package beside the hyphenated ops-report directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops-report"))
from opsreport.proposals import ParsedProposal, parse_report  # noqa: E402

if TYPE_CHECKING:
    from .github import GitHub
    from .session import ParentSession

EVENTS = {"proposal_pending", "proposal_received"}


def source_key(source: ProposalSource) -> tuple[int, str]:
    return source.report_number, source.proposal_id


def event_source(event: dict[str, Any]) -> ProposalSource:
    raw = object_value(event.get("source"))
    if set(raw) != {"proposal_id", "report_number", "option"}:
        raise ValueError("invalid proposal event source")
    return ProposalSource(**raw)


def read_proposal(gh: GitHub, source: ProposalSource, parent: int) -> ParsedProposal:
    report = gh.issue(source.report_number)
    if "ops-report" not in {label["name"] for label in report["labels"]}:
        raise ValueError("proposal source is not an ops-report issue")
    meta, proposals = parse_report(report["body"])
    if report["state"].lower() != "open" or meta.status != "complete":
        raise ValueError("proposal report must be open and complete")
    prop = next((p for p in proposals if p.pid == source.proposal_id), None)
    if prop is None:
        raise ValueError("proposal no longer exists in the report")
    prop.requirements()
    if prop.parent() != parent:
        raise ValueError("proposal does not name this parent; assign it explicitly")
    if prop.selection() != (True, source.option) or prop.deferred_reason():
        raise ValueError("proposal adoption/option changed; reconcile before retry")
    return prop


def record(source: ProposalSource, prop: ParsedProposal) -> dict[str, Any]:
    return {
        "source": asdict(source),
        "proposal_sha256": digest(prop.snapshot()),
        "adoption": "selected",
    }


def matching(
    history: list[dict[str, Any]], source: ProposalSource, expected: dict[str, Any]
) -> list[dict[str, Any]]:
    found = []
    for event in history:
        if event.get("event") not in EVENTS:
            continue
        if source_key(event_source(event)) != source_key(source):
            continue
        if any(event.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"{source.proposal_id}: delivered adoption/option/proposal changed; reconcile by hand"
            )
        found.append(event)
    return found


def deliver(
    session: ParentSession, source: ProposalSource, prop: ParsedProposal
) -> str:
    expected = record(source, prop)
    for key in session.store.parents:
        repo, _, other = key.rpartition("#")
        if repo == session.gh.repo.casefold() and int(other) != session.parent:
            history = events(session.gh.comments(int(other)), session.login, int(other))
            if matching(history, source, expected):
                raise ValueError(
                    "proposal already delivered to another parent; reconcile by hand"
                )
    found = matching(session.history(), source, expected)
    if any(e["event"] == "proposal_received" for e in found):
        return "受領済み（要件合意・起票は別途）"
    if session.row.get("status") != "active":
        return "未割当（担当親セッション未登録・停止中）"
    if session.gh.issue(session.parent)["state"].lower() != "open":
        return "未割当（親issue停止中）"
    # Re-read adoption immediately before publishing a new pending record.
    current = read_proposal(session.gh, source, session.parent)
    if record(source, current) != expected:
        raise ValueError("proposal changed during delivery")
    session.post(
        {"event": "proposal_pending", **expected},
        f"採用済み提案の受領待ち: #{source.report_number} / `{source.proposal_id}`\n\n"
        f"採用の記録: 方向性を採用。選択肢: {source.option or 'なし'}。\n\n"
        "担当親が receive-proposal で要件確認の対象として受領してください。受領だけでは子issue化・実装を始めません。",
    )
    return "採用済み提案の受領待ち"


def selected_source(session: ParentSession, report: int, pid: str) -> ProposalSource:
    report = number(report)
    raw = session.gh.issue(report)
    _, parsed = parse_report(raw["body"])
    prop = next((p for p in parsed if p.pid == pid), None)
    if prop is None:
        raise ValueError("proposal not found")
    return ProposalSource(pid, report, prop.selection()[1])


def receive(session: ParentSession, report: int, pid: str) -> dict[str, Any]:
    session.owner()
    if session.gh.issue(session.parent)["state"].lower() != "open":
        raise ValueError("cannot receive a proposal for a closed parent")
    source = selected_source(session, report, pid)
    prop = read_proposal(session.gh, source, session.parent)
    expected = record(source, prop)
    found = matching(session.history(), source, expected)
    reused = any(e["event"] == "proposal_received" for e in found)
    deliver(session, source, prop)
    if not reused:
        if (
            record(source, read_proposal(session.gh, source, session.parent))
            != expected
        ):
            raise ValueError("proposal changed before receipt")
        session.post(
            {"event": "proposal_received", **expected},
            f"採用済み提案を要件確認の対象として受領: #{report} / `{pid}`。\n\n"
            "人間との別途の要件合意を記録してから分割・起票・委任します。",
        )
    if record(source, read_proposal(session.gh, source, session.parent)) != expected:
        raise ValueError("proposal changed during receipt")
    return {"parent": session.parent, **expected, "received": True, "reused": reused}


def receipts(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[tuple[int, str], dict[str, Any]] = {}
    for event in history:
        if event.get("event") == "proposal_received":
            source = event_source(event)
            value = {
                key: event[key] for key in ("source", "proposal_sha256", "adoption")
            }
            key = source_key(source)
            if key in by_source and by_source[key] != value:
                raise ValueError("conflicting proposal receipts")
            by_source[key] = value
    return [by_source[key] for key in sorted(by_source)]


def agreed_receipts(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agreed: dict[tuple[int, str], dict[str, Any]] = {}
    for event in history:
        if event.get("event") == "checking":
            agreed.clear()
        elif event.get("event") == "proposal_agreed":
            source = event_source(event)
            value = {
                key: event[key] for key in ("source", "proposal_sha256", "adoption")
            }
            agreed[source_key(source)] = value
    return [agreed[key] for key in sorted(agreed)]


def require_receipt(session: ParentSession, source: ProposalSource) -> dict[str, Any]:
    value = record(source, read_proposal(session.gh, source, session.parent))
    history = session.history()
    matching(history, source, value)
    if value not in receipts(history):
        raise ValueError(
            "proposal requires explicit receipt before requirements agreement"
        )
    return value


def require_source(session: ParentSession, source: ProposalSource) -> ParsedProposal:
    from .adopted_tasks import require_task_source

    automatic = require_task_source(session, source)
    if automatic is not None:
        return automatic
    prop = read_proposal(session.gh, source, session.parent)
    value = record(source, prop)
    history = session.history()
    if value not in receipts(history) or value not in agreed_receipts(history):
        raise ValueError(
            "proposal requires explicit receipt and a separate human requirements agreement"
        )
    matching(history, source, value)
    return prop
