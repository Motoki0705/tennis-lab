"""Turn checked proposals of open report issues into ``ai-proposed`` child issues.

Idempotency: a child issue's title starts with ``[<proposal ID>]``. Before creating
anything we scan every issue title (open and closed), so a re-run -- including one
after a crash between "create child" and "annotate report" -- links the existing
child instead of filing a duplicate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import shell
from .collect import AI_PROPOSED_LABEL, REPORT_LABEL
from .proposals import (
    CHILD_TITLE_RE,
    ParsedProposal,
    SelectionError,
    annotate_children,
    parse_proposals,
)
from .shell import CommandError

LOG = logging.getLogger("weekly_report.triage")
MAX_TITLE = 200
_ISSUE_URL_RE = re.compile(r"/issues/(\d+)\s*$")

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
    title = re.sub(r"（優先度: .）$", "", prop.title).strip()
    if option:
        title = f"{title}（選択: {option}）"
    full = f"[{prop.pid}] {title}"
    return full if len(full) <= MAX_TITLE else full[: MAX_TITLE - 1] + "…"


def child_body(prop: ParsedProposal, option: str | None, report_number: int) -> str:
    head = re.sub(r"^- (\[[ xX]\] )?", "- ", prop.block[0])
    rest = [line[2:] if line.startswith("  ") else line for line in prop.block[1:]]
    lines = [
        f"<!-- ops-report-proposal: {prop.pid} -->",
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
        number = int(report["number"])
        to_link: dict[str, int] = {}
        for prop in parse_proposals(report["body"]):
            if prop.child is not None:
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
                to_link[prop.pid] = children[prop.pid]
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
            to_link[prop.pid] = child
            result.created.append((prop.pid, child))
            LOG.info("%s: created child issue #%d", prop.pid, child)
        if not to_link or dry_run:
            continue
        # Re-read right before editing so concurrent checkbox edits are not lost.
        fresh = shell.gh_json(
            ["issue", "view", str(number), "--json", "body"], cwd=repo_root
        )
        new_body = annotate_children(fresh["body"], to_link)
        if new_body == fresh["body"]:
            continue
        body_file = work_dir / f"report-{number}.md"
        body_file.write_text(new_body, encoding="utf-8")
        shell.gh(
            ["issue", "edit", str(number), "--body-file", str(body_file)], cwd=repo_root
        )
        LOG.info("#%d: annotated %d proposal(s)", number, len(to_link))
    return result
