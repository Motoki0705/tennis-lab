"""Turn checked proposals of open report issues into ``ai-proposed`` child issues.

Idempotency: a child issue's title starts with ``[<proposal ID>]``. Before creating
anything we scan every issue title (open and closed), so a re-run -- including one
after a crash between "create child" and "annotate report" -- links the existing
child instead of filing a duplicate. An existing child is linked only if its body
marker names the same proposal ID *and* report issue; anything else is an error,
never a silent relink to an unrelated issue.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from . import shell
from .collect import AI_PROPOSED_LABEL, REPORT_LABEL
from .proposals import (
    CHILD_TITLE_RE,
    ID_PATTERN,
    ParsedProposal,
    SelectionError,
    annotate_children,
    parse_report,
    unescape_head_text,
)
from .shell import CommandError

LOG = logging.getLogger("weekly_report.triage")
MAX_TITLE = 200
_ISSUE_URL_RE = re.compile(r"/issues/(\d+)\s*$")
# ``option`` is the percent-encoded selected option ("" = proposal without options),
# so a reused child can be checked against the report's current selection.
_CHILD_MARKER_RE = re.compile(
    rf"^<!-- ops-report-proposal: (?P<id>{ID_PATTERN}) report=#(?P<report>\d+)"
    r" option=(?P<option>\S*) -->$"
)


def child_marker(pid: str, report_number: int, option: str | None) -> str:
    encoded = quote(option or "", safe="")
    return (
        f"<!-- ops-report-proposal: {pid} report=#{report_number} option={encoded} -->"
    )


LABEL_SPECS: dict[str, tuple[str, str]] = {
    REPORT_LABEL: ("0E8A16", "週次運用レポート（.agents/ops/weekly-report）"),
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
        LOG.info("created label %s", name)


def existing_children(repo_root: Path) -> dict[str, int]:
    rows = shell.gh_list(
        ["issue", "list", "--state", "all", "--json", "number,title"],
        limit=20_000,
        cwd=repo_root,
    )
    children: dict[str, int] = {}
    for row in rows:
        m = CHILD_TITLE_RE.match(row["title"])
        if not m:
            continue
        pid = m.group(1)
        if pid in children:
            raise CommandError(
                f"proposal {pid} already has two child issues (#{children[pid]}, #{row['number']}); "
                "resolve the duplicate by hand"
            )
        children[pid] = int(row["number"])
    return children


def child_title(prop: ParsedProposal, option: str | None) -> str:
    title = unescape_head_text(re.sub(r"（優先度: .）$", "", prop.title).strip())
    if option:
        title = f"{title}（選択: {option}）"
    full = f"[{prop.pid}] {title}"
    return full if len(full) <= MAX_TITLE else full[: MAX_TITLE - 1] + "…"


def child_body(prop: ParsedProposal, option: str | None, report_number: int) -> str:
    head = re.sub(r"^- (\[[ xX]\] )?", "- ", prop.block[0])
    rest = [line[2:] if line.startswith("  ") else line for line in prop.block[1:]]
    lines = [
        child_marker(prop.pid, report_number, option),
        f"週次運用レポート #{report_number} の提案 `{prop.pid}` から日次triageが起票。",
        "",
        "## 提案",
        "",
        head,
        *(f"  {line}" for line in rest),
        "",
    ]
    if option:
        lines += [f"選択された対応: **{option}**", ""]
    lines += ["## 出典", "", f"- レポート: #{report_number}", ""]
    return "\n".join(lines)


@dataclass
class TriageResult:
    created: list[tuple[str, int]] = field(default_factory=list)
    linked_existing: list[tuple[str, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _create_issue(title: str, body: str, *, repo_root: Path, work_dir: Path) -> int:
    body_file = work_dir / "child-body.md"
    body_file.write_text(body, encoding="utf-8")
    out = shell.gh(
        ["issue", "create", "--title", title, "--body-file", str(body_file),
         "--label", AI_PROPOSED_LABEL],
        cwd=repo_root,
    )  # fmt: skip
    m = _ISSUE_URL_RE.search(out.strip())
    if not m:
        raise CommandError(f"gh issue create printed no issue URL: {out!r}")
    return int(m.group(1))


def verify_child(
    pid: str, child: int, report_number: int, repo_root: Path
) -> str | None:
    """Raise ValueError unless issue ``child`` was filed for ``pid`` of this report;
    return the option recorded in its marker (None for a proposal without options)."""
    body = shell.gh_json(
        ["issue", "view", str(child), "--json", "body"], cwd=repo_root
    )["body"]
    m = _CHILD_MARKER_RE.match(body.split("\n", 1)[0].rstrip("\r"))
    if not m or m.group("id") != pid or int(m.group("report")) != report_number:
        found = f"{m.group('id')} of #{m.group('report')}" if m else "no marker"
        raise ValueError(
            f"{pid}: existing issue #{child} is not this proposal's child "
            f"(expected {pid} of #{report_number}, found {found}); resolve by hand"
        )
    return unquote(m.group("option")) or None


def check_selection(prop: ParsedProposal, child: int, recorded: str | None) -> None:
    """Raise ValueError unless ``prop`` is still selected with the option ``child``
    was filed for (a human may change the checkboxes at any time)."""
    selected, option = prop.selection()  # SelectionError is a ValueError
    if not selected:
        raise ValueError(
            f"{prop.pid}: child #{child} was filed for option {recorded!r} but the "
            "proposal is no longer selected; resolve by hand"
        )
    if option != recorded:
        raise ValueError(
            f"{prop.pid}: existing child #{child} was filed for option {recorded!r} "
            f"but the report now selects {option!r}; resolve by hand "
            "(close the child or restore the selection)"
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
    if not reports:
        LOG.info("no open %s issues", REPORT_LABEL)
        return result
    ensure_labels([AI_PROPOSED_LABEL], repo_root=repo_root, dry_run=dry_run)
    children = existing_children(repo_root)
    for report in sorted(reports, key=lambda r: int(r["number"])):
        try:
            _triage_report(report, children, result, repo_root, work_dir, dry_run)
        except ValueError as exc:  # malformed body; other reports still proceed
            result.errors.append(f"#{report['number']} {exc}")
            LOG.error("#%s %s", report["number"], exc)
    return result


def _triage_report(
    report: dict[str, Any],
    children: dict[str, int],
    result: TriageResult,
    repo_root: Path,
    work_dir: Path,
    dry_run: bool,
) -> None:
    number = int(report["number"])
    _, parsed = parse_report(report["body"])
    # proposal ID -> (child issue, option the child was filed for)
    to_link: dict[str, tuple[int, str | None]] = {}
    for prop in parsed:
        if prop.child is not None:
            # A linked "→ #N" must still be backed by a child filed for this proposal
            # and the option selected now (forged notes, later checkbox changes).
            try:
                recorded = verify_child(prop.pid, prop.child, number, repo_root)
                check_selection(prop, prop.child, recorded)
            except ValueError as exc:
                result.errors.append(f"#{number} {exc}")
                LOG.error("#%d %s", number, exc)
            continue
        try:
            selected, option = prop.selection()
        except SelectionError as exc:
            result.errors.append(f"#{number} {exc}")
            LOG.error("#%d %s", number, exc)
            continue
        if not selected:
            continue
        if prop.pid in children:
            try:
                # e.g. the report edit failed after creating a "close" child and the
                # human then switched to "継続": never relink the stale child.
                recorded = verify_child(prop.pid, children[prop.pid], number, repo_root)
                check_selection(prop, children[prop.pid], recorded)
            except ValueError as exc:
                result.errors.append(f"#{number} {exc}")
                LOG.error("#%d %s", number, exc)
                continue
            to_link[prop.pid] = (children[prop.pid], recorded)
            result.linked_existing.append((prop.pid, children[prop.pid]))
            LOG.info(
                "%s: child #%d already exists, linking",
                prop.pid,
                children[prop.pid],
            )
            continue
        title = child_title(prop, option)
        if dry_run:
            LOG.info("dry-run: would create child issue %r", title)
            continue
        child = _create_issue(
            title,
            child_body(prop, option, number),
            repo_root=repo_root,
            work_dir=work_dir,
        )
        children[prop.pid] = child
        to_link[prop.pid] = (child, option)
        result.created.append((prop.pid, child))
        LOG.info("%s: created child issue #%d", prop.pid, child)
    if not to_link or dry_run:
        return
    # Re-read right before editing so concurrent checkbox edits are not lost.
    fresh = shell.gh_json(
        ["issue", "view", str(number), "--json", "body"], cwd=repo_root
    )
    # The human may have changed the checkboxes while children were being filed.
    _, current = parse_report(fresh["body"])
    annotate: dict[str, int] = {}
    for prop in current:
        if prop.pid not in to_link:
            continue
        child, recorded = to_link[prop.pid]
        try:
            check_selection(prop, child, recorded)
        except ValueError as exc:
            result.errors.append(f"#{number} {exc}")
            LOG.error("#%d %s (not annotated)", number, exc)
            continue
        annotate[prop.pid] = child
    for pid in sorted(set(to_link) - {prop.pid for prop in current}):
        msg = f"{pid}: child #{to_link[pid][0]} exists but the proposal vanished from the report"
        result.errors.append(f"#{number} {msg}")
        LOG.error("#%d %s", number, msg)
    new_body = annotate_children(fresh["body"], annotate)
    if new_body == fresh["body"]:
        return
    body_file = work_dir / f"report-{number}.md"
    body_file.write_text(new_body, encoding="utf-8")
    shell.gh(
        ["issue", "edit", str(number), "--body-file", str(body_file)], cwd=repo_root
    )
    LOG.info("#%d: annotated %d proposal(s)", number, len(annotate))
