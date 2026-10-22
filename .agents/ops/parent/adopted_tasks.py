"""Validate triage provenance under a separately recorded autonomous basis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from issues import (
    ProposalSource,
    index_proposals,
    parse_proposal_source,
    parse_task_metadata,
)

from . import execution
from .policy import digest

if TYPE_CHECKING:
    from opsreport.proposals import ParsedProposal

    from .github import GitHub
    from .session import ParentSession


def check_task_adoptions(gh: GitHub, body: str) -> list[ParsedProposal]:
    """Check every context source, including tasks without a single index anchor.

    The development workflow calls this for both parent and task immediately before
    execution and merge, in addition to validating its ownership/execution basis.
    This function grants no authority and writes no state.
    """
    metadata = parse_task_metadata(body)
    if metadata is None:
        return []
    from .proposals import parse_report

    checked = []
    for binding in metadata["bindings"]:
        source = ProposalSource(**binding["source"])
        report = gh.issue(source.report_number)
        meta, parsed = parse_report(report["body"])
        if (
            report["state"] != "open"
            or meta.status != "complete"
            or "ops-report" not in {label["name"] for label in report["labels"]}
        ):
            raise ValueError(
                "adopted task's source report must remain open and complete"
            )
        prop = next((p for p in parsed if p.pid == source.proposal_id), None)
        if (
            prop is None
            or prop.selection() != (True, source.option)
            or prop.deferred_reason()
        ):
            raise ValueError(
                "task adoption/option changed; stop and reconcile the source"
            )
        prop.requirements()
        if digest(prop.snapshot()) != binding["proposal_sha256"]:
            raise ValueError(
                "task proposal digest changed; an old adoption cannot authorize new content"
            )
        checked.append(prop)
    return checked


def require_task_source(
    session: ParentSession, source: ProposalSource
) -> ParsedProposal | None:
    rows = session.gh.issue_rows()
    indexed = index_proposals((r["number"], r.get("body") or "") for r in rows)
    n = indexed.get((source.report_number, source.proposal_id))
    if n is None:
        return None
    issue = session.gh.issue(n)
    metadata = parse_task_metadata(issue["body"])
    if metadata is None:
        return None  # The explicit human receipt/agreement path remains separate.
    if parse_proposal_source(issue["body"]) != source:
        raise ValueError("task source/option does not match the requested adoption")
    if execution.current_basis(session)["kind"] != "autonomous_decision":
        raise ValueError(
            "triage task requires a separately recorded autonomous execution basis"
        )
    family = session.gh.family(session.parent)
    if n != session.parent and (n not in family or family[n].parent != session.parent):
        raise ValueError("adopted task is not managed by this parent")
    check_task_adoptions(session.gh, session.gh.issue(session.parent)["body"])
    checked = check_task_adoptions(session.gh, issue["body"])
    return next(
        p
        for p, binding in zip(checked, metadata["bindings"], strict=True)
        if ProposalSource(**binding["source"]) == source
    )
