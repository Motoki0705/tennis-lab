"""Record adoption receipts and materialize unowned, adopted proposals as tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from issues import (
    ProposalSource,
    index_proposals,
    parse_proposal_source,
    parse_task_metadata,
)
from parent.github import GitHub
from parent.proposals import deliver, matching, record
from parent.session import ParentSession
from parent.state import Store, state_directory

from . import organization_apply, shell
from .collect import AI_PROPOSED_LABEL, REPORT_LABEL
from .proposals import (
    ParsedProposal,
    annotate_children,
    annotate_handoffs,
    parse_report,
)

LOG = logging.getLogger("ops_report.triage")
LABEL_SPECS: dict[str, tuple[str, str]] = {
    REPORT_LABEL: ("0E8A16", "運用レポート（.agents/ops/ops-report）"),
    AI_PROPOSED_LABEL: ("C5DEF5", "AIが起票した提案・派生課題"),
}


def ensure_labels(names: list[str], *, repo_root: Path, dry_run: bool) -> None:
    existing = {
        row["name"]
        for row in shell.gh_list(
            ["label", "list", "--json", "name"], limit=1000, cwd=repo_root
        )
    }
    for name in names:
        if name in existing:
            continue
        if dry_run:
            LOG.info("dry-run: would create label %s", name)
            continue
        color, description = LABEL_SPECS[name]
        shell.gh(
            ["label", "create", name, "--color", color, "--description", description],
            cwd=repo_root,
        )


@dataclass
class TriageResult:
    pending: list[tuple[str, int]] = field(default_factory=list)
    received: list[tuple[str, int]] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)
    linked_existing: list[tuple[str, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def memory_decisions(
    history: dict[str, Any], issues: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """GitHub is authoritative; unchecked alone means undecided, never deferred."""
    children = index_proposals(
        [(int(row["number"]), row["body"] or "") for row in issues]
    )
    bodies = {int(row["number"]): row["body"] or "" for row in issues}
    decisions = []
    for report in history["reports"]:
        meta, parsed = parse_report(report["body"])
        if meta.status != "complete":
            continue
        for prop in parsed:
            selected, option = prop.selection()
            reason = prop.deferred_reason()
            if selected and reason:
                raise ValueError(f"{prop.pid}: both adopted and deferred")
            child = children.get((int(report["number"]), prop.pid))
            automatic = (
                child is not None and parse_task_metadata(bodies[child]) is not None
            )
            if prop.child is not None and prop.child != child:
                raise ValueError(
                    f"{prop.pid}: child annotation does not match proposal metadata"
                )
            if child is not None and not selected and not automatic:
                raise ValueError(f"{prop.pid}: existing child but adoption revoked")
            if not selected and not reason:
                continue
            refs = f"{report['url']} / {prop.pid}"
            if child is not None:
                if (selected or not automatic) and parse_proposal_source(
                    bodies[child]
                ) != ProposalSource(prop.pid, int(report["number"]), option):
                    raise ValueError(f"{prop.pid}: existing child option mismatch")
                refs += f" / 子issue #{child}"
            parent = prop.parent()
            if parent:
                refs += f" / 親issue #{parent}"
            decisions.append(
                {
                    "proposal_id": prop.pid,
                    "decision": "selected" if selected else "deferred",
                    "reason": f"方向性を採用（選択肢: {option or 'なし'}）。要件合意は別途。"
                    if selected
                    else str(reason),
                    "source": refs,
                }
            )
    return decisions


def collect_decisions(repo_root: Path, history: dict[str, Any]) -> list[dict[str, str]]:
    rows = shell.gh_list(
        ["issue", "list", "--state", "all", "--json", "number,body"],
        limit=20_000,
        cwd=repo_root,
    )
    return memory_decisions(history, rows)


def _verify_child(gh: GitHub, source: ProposalSource, child: int) -> None:
    if parse_proposal_source(gh.issue(child)["body"]) != source:
        raise ValueError(
            f"{source.proposal_id}: existing issue #{child} source/option mismatch"
        )


def triage(*, repo_root: Path, work_dir: Path, dry_run: bool) -> TriageResult:
    result = TriageResult()
    reports = shell.gh_list(
        [
            "issue",
            "list",
            "--label",
            REPORT_LABEL,
            "--state",
            "open",
            "--json",
            "number,title,body",
        ],
        limit=500,
        cwd=repo_root,
    )
    # Validate all report versions before any mutation, including mixed v1/v2.
    parsed_reports = []
    for report in reports:
        try:
            meta, parsed = parse_report(report["body"])
            if meta.status == "complete":
                parsed_reports.append((report, parsed))
        except ValueError as exc:
            result.errors.append(f"#{report['number']} {exc}")
    if result.errors or not parsed_reports:
        return result
    rows = shell.gh_list(
        ["issue", "list", "--state", "all", "--json", "number,body"],
        limit=20_000,
        cwd=repo_root,
    )
    children = index_proposals([(int(r["number"]), r["body"] or "") for r in rows])
    repo = shell.gh_json(["repo", "view", "--json", "nameWithOwner"], cwd=repo_root)[
        "nameWithOwner"
    ]
    gh = GitHub(repo)
    # Use exactly the parent entry's registry lock, without impersonating an owner.
    with Store(state_directory()).locked() as store:
        sessions = {
            int(key.rsplit("#", 1)[1]): ParentSession(
                gh, store, int(key.rsplit("#", 1)[1]), None
            )
            for key in store.parents
            if key.rsplit("#", 1)[0] == repo.casefold()
        }
        automatic: list[tuple[int, ParsedProposal]] = []
        for report, parsed in parsed_reports:
            _triage_report(
                report,
                parsed,
                children,
                result,
                gh,
                store,
                sessions,
                work_dir,
                dry_run,
                automatic,
            )
        if automatic:
            organization_apply.run(
                gh=gh,
                store=store,
                root=repo_root,
                work_dir=work_dir,
                candidates=automatic,
                dry_run=dry_run,
                result=result,
            )
    return result


def _triage_report(
    report: dict[str, Any],
    parsed: list[ParsedProposal],
    children: dict[tuple[int, str], int],
    result: TriageResult,
    gh: GitHub,
    store: Store,
    sessions: dict[int, ParentSession],
    work_dir: Path,
    dry_run: bool,
    automatic: list[tuple[int, ParsedProposal]],
) -> None:
    number = int(report["number"])
    statuses: dict[str, str] = {}
    linked: dict[str, int] = {}
    expectations: dict[str, dict[str, Any]] = {}
    for prop in parsed:
        try:
            selected, option = prop.selection()
            if selected and prop.deferred_reason():
                raise ValueError(f"{prop.pid}: both adopted and deferred")
            parent = prop.parent()
            source = ProposalSource(prop.pid, number, option)
            expected = record(source, prop)
            if parent is not None and parent not in sessions:
                sessions[parent] = ParentSession(gh, store, parent, None)
            previous = []
            for session in sessions.values():
                previous += matching(session.history(), source, expected)
            child = children.get((number, prop.pid))
            if prop.child is not None and prop.child != child:
                raise ValueError(
                    f"{prop.pid}: forged or mismatched child annotation #{prop.child}"
                )
            managed = (
                child is not None
                and parse_task_metadata(gh.issue(child)["body"]) is not None
            )
            # An explicit existing owner keeps the established receipt workflow.
            # Unowned directions need no manual parent or fabricated human consent.
            if managed or (
                child is None
                and not previous
                and (parent is None or not sessions[parent].row)
            ):
                if selected:
                    try:
                        prop.requirements()
                    except ValueError as exc:
                        result.errors.append(f"#{number} {exc}")
                        expectations[prop.pid] = expected
                        statuses[prop.pid] = (
                            f"保留: {exc}。根拠: レポート #{number} の必須項目。再開条件: 目的・範囲・受入条件など不足項目を追記する。次の対応: 提案を具体化してtriageを再実行。"
                        )
                        continue
                if selected or managed:
                    automatic.append((number, prop))
                continue
            if not selected:
                if child or previous:
                    raise ValueError(
                        f"{prop.pid}: adoption revoked after delivery/child creation"
                    )
                continue
            prop.requirements()
            expectations[prop.pid] = expected
            if child is not None:
                _verify_child(gh, source, child)
                linked[prop.pid] = child
                result.linked_existing.append((prop.pid, child))
                statuses[prop.pid] = "既存子issueを再利用"
            elif any(e["parent"] != parent for e in previous):
                raise ValueError(
                    f"{prop.pid}: already delivered to another parent; reconcile by hand"
                )
            elif parent is None:
                statuses[prop.pid] = "未割当（関連親issue未指定）"
                result.unassigned.append(prop.pid)
            else:
                if dry_run:
                    LOG.info(
                        "dry-run: would deliver %s to parent #%d", prop.pid, parent
                    )
                    continue
                status = deliver(sessions[parent], source, prop)
                statuses[prop.pid] = f"親 #{parent}: {status}"
                if status.startswith("未割当"):
                    statuses[prop.pid] += (
                        "。根拠: 既存parent registryと親issueの状態。再開条件: 担当親の明示再開・引継ぎを確認。次の対応: 既存担当が採用提案を照合する。"
                    )
                    result.unassigned.append(prop.pid)
                elif status.startswith("受領済み"):
                    result.received.append((prop.pid, parent))
                else:
                    result.pending.append((prop.pid, parent))
            LOG.info("%s: %s", prop.pid, statuses[prop.pid])
        except ValueError as exc:
            result.errors.append(f"#{number} {exc}")
            LOG.error("#%d %s", number, exc)
    if dry_run or not statuses:
        return
    fresh = gh.issue(number)
    meta, current = parse_report(fresh["body"])
    if fresh["state"].lower() != "open" or meta.status != "complete":
        raise ValueError("report closed or incomplete during triage")
    by_id = {p.pid: p for p in current}
    for pid in list(statuses):
        fresh_prop = by_id.get(pid)
        old = next(p for p in parsed if p.pid == pid)
        if (
            fresh_prop is None
            or fresh_prop.selection() != old.selection()
            or fresh_prop.parent() != old.parent()
            or fresh_prop.deferred_reason()
            or record(
                ProposalSource(pid, number, fresh_prop.selection()[1]), fresh_prop
            )
            != expectations[pid]
        ):
            result.errors.append(
                f"#{number} {pid}: proposal/adoption changed during triage; not annotated"
            )
            statuses.pop(pid)
            linked.pop(pid, None)
            continue
        if pid in linked:
            _verify_child(
                gh, ProposalSource(pid, number, fresh_prop.selection()[1]), linked[pid]
            )
    body = annotate_handoffs(annotate_children(fresh["body"], linked), statuses)
    if body != fresh["body"]:
        (work_dir / f"report-{number}.md").write_text(body, encoding="utf-8")
        if gh.issue(number)["body"] != fresh["body"]:
            raise ValueError("report changed before annotation; retry triage")
        gh.write("PATCH", f"repos/{gh.repo}/issues/{number}", {"body": body})
