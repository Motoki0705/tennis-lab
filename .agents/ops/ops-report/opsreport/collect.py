"""Pin the report boundary and dedicated memory; research is fetched on demand."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import memory, proposals, shell
from .shell import CommandError

REPORT_LABEL = "ops-report"
AI_PROPOSED_LABEL = "ai-proposed"
FETCH_TIMEOUT = 600


@dataclass(frozen=True)
class BaseRef:
    """The base ref as given and the commit pinned for the whole run."""

    name: str
    sha: str
    # "<remote> <branch>" when a remote-tracking ref was fetched first; None for a
    # local ref / SHA, which is used as is (recorded in the report, not hidden).
    fetched_from: str | None


def pin_base(repo_root: Path, name: str) -> BaseRef:
    """Fetch ``name`` if it is a remote-tracking ref, then pin its commit SHA.

    A stale local ``origin/main`` would silently drop commits while the period still
    advances, so a failed fetch is an error.
    """
    full = shell.git(["rev-parse", "--symbolic-full-name", name], cwd=repo_root).strip()
    fetched_from = None
    if full.startswith("refs/remotes/"):
        tracking = full.removeprefix("refs/remotes/")
        remotes = shell.git(["remote"], cwd=repo_root).split()
        remote = max(
            (r for r in remotes if tracking.startswith(r + "/")), key=len, default=None
        )
        if remote is None:
            raise CommandError(
                f"{name} ({full}) does not belong to a configured remote"
            )
        branch = tracking.removeprefix(remote + "/")
        shell.run_process_group(
            ["git", "fetch", "--no-tags", remote, f"+refs/heads/{branch}:{full}"],
            cwd=repo_root,
            timeout=FETCH_TIMEOUT,
        )
        fetched_from = f"{remote} {branch}"
    sha = shell.git(
        ["rev-parse", "--verify", f"{name}^{{commit}}"], cwd=repo_root
    ).strip()
    return BaseRef(name=name, sha=sha, fetched_from=fetched_from)


@dataclass(frozen=True)
class CollectConfig:
    repo_root: Path
    now: datetime
    base: BaseRef

    @property
    def base_ref(self) -> str:
        return self.base.name

    @property
    def rev(self) -> str:
        """What every git query reads: the pinned SHA, never the moving ref."""
        return self.base.sha


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def collect_report_history(repo_root: Path) -> dict[str, Any]:
    rows = shell.gh_list(
        [
            "issue", "list", "--label", REPORT_LABEL, "--state", "all",
            "--json", "number,title,state,createdAt,body,url,author",
        ],
        limit=500,
        cwd=repo_root,
    )  # fmt: skip
    rows.sort(key=lambda r: parse_ts(r["createdAt"]))
    summarized = []
    for i, row in enumerate(rows):
        try:
            meta, parsed = proposals.parse_report(row["body"])
        except ValueError as exc:
            raise CommandError(
                f"{REPORT_LABEL} issue #{row['number']} has a malformed body: {exc}"
            ) from exc
        item: dict[str, Any] = {
            k: row[k] for k in ("number", "title", "state", "createdAt", "url")
        }
        item["slot"] = meta.slot
        item["author"] = row["author"]["login"]
        item["body_sha256"] = hashlib.sha256(row["body"].encode("utf-8")).hexdigest()
        item["period_start"] = meta.since.isoformat()
        item["period_end"] = meta.until.isoformat()
        item["status"] = meta.status
        item["base_sha"] = meta.base
        item["audit"] = meta.audit
        # The body is compared again under the issue lock before replacing an
        # incomplete report. Human selections must never be overwritten.
        item["body"] = row["body"]
        # Proposal status only for the two most recent reports to bound prompt size.
        if i >= len(rows) - 2:
            item["proposals"] = [_proposal_status(p) for p in parsed]
        summarized.append(item)
    slots = [r["slot"] for r in summarized]
    if len(slots) != len(set(slots)):
        raise CommandError(
            "multiple ops-report issues for the same slot; resolve duplicates by hand"
        )
    summarized.sort(key=lambda r: parse_ts(r["period_end"]))
    return {"reports": summarized}


def _proposal_status(p: proposals.ParsedProposal) -> dict[str, Any]:
    try:
        selected, option = p.selection()
        state = (
            "selected"
            if selected
            else "deferred"
            if p.deferred_reason()
            else "unselected"
        )
    except proposals.SelectionError as exc:
        state, option = f"ambiguous: {exc}", None
    return {
        "id": p.pid,
        "title": p.title,
        "state": state,
        "option": option,
        "child_issue": p.child,
        "deferred_reason": p.deferred_reason(),
    }


def collect_all(
    cfg: CollectConfig,
    *,
    history: dict[str, Any],
    initial_since: datetime,
) -> dict[str, Any]:
    """Freeze [previous completed boundary, now), never an inventory or logs."""
    reports = [r for r in history["reports"] if r["status"] == "complete"]
    since = (
        datetime.fromisoformat(reports[-1]["period_end"]).astimezone(cfg.now.tzinfo)
        if reports
        else initial_since
    )
    if since > cfg.now:
        raise CommandError("previous report boundary is after this run's time")
    repository = shell.gh_json(
        ["repo", "view", "--json", "nameWithOwner"], cwd=cfg.repo_root
    )["nameWithOwner"]
    return {
        "meta": {
            "generated_at": cfg.now.isoformat(timespec="seconds"),
            "period_start": since.isoformat(timespec="seconds"),
            "period_end": cfg.now.isoformat(timespec="seconds"),
            "base_ref": cfg.base_ref,
            "base_sha": cfg.rev,
            "base_fetched_from": cfg.base.fetched_from,
            "repository": repository,
        },
        "ops_memory": memory.collect(cfg.repo_root, cfg.rev),
    }
