"""Deterministic data collection for the weekly report.

Everything here is computed from git / gh / the filesystem without an agent, so
the same repository state always yields the same facts. The agent only interprets
this data and reads the codebase (read-only) for deeper analysis.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import proposals, shell
from .shell import CommandError

REPORT_LABEL = "ops-report"
AI_PROPOSED_LABEL = "ai-proposed"
DEFAULT_PERIOD_DAYS = 7
CLEANUP_DIR = Path(".agents/ops/cleanup")
MEMORY_DIR = Path(".agents/memory")
KNOWLEDGE_SUMMARY = Path("knowledge/summary.md")
CODE_SUFFIXES = ("*.py", "*.sh", "*.ts", "*.tsx", "*.js")
# Vendored / bundled third-party code would dominate size and TODO statistics.
EXCLUDED_PATHSPECS = (
    ":!third_party",
    ":!**/vendor/**",
    ":!**/three.*.js",
    ":!**/*.min.js",
)
MAX_TEXT = 20_000

_ISSUE_FIELDS = "number,title,labels,createdAt,updatedAt,author"
_PR_FIELDS = "number,title,headRefName,author,createdAt,updatedAt,isDraft,labels,url"


@dataclass(frozen=True)
class CollectConfig:
    repo_root: Path
    now: datetime
    base_ref: str
    stale_pr_days: int
    cleanup_cmd: tuple[str, ...] | None


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _names(labels: list[dict[str, Any]]) -> list[str]:
    return sorted(str(label["name"]) for label in labels)


def _truncate(text: str, limit: int = MAX_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated: {len(text) - limit} more chars]"


def _days(now: datetime, ts: str) -> int:
    return (now - parse_ts(ts)).days


def collect_report_history(cfg: CollectConfig) -> dict[str, Any]:
    rows = shell.gh_list(
        [
            "issue", "list", "--label", REPORT_LABEL, "--state", "all",
            "--json", "number,title,state,createdAt,body,url",
        ],
        limit=500,
        cwd=cfg.repo_root,
    )  # fmt: skip
    rows.sort(key=lambda r: parse_ts(r["createdAt"]))
    summarized = []
    for i, row in enumerate(rows):
        item: dict[str, Any] = {
            k: row[k] for k in ("number", "title", "state", "createdAt", "url")
        }
        # Proposal status only for the two most recent reports to bound prompt size.
        if i >= len(rows) - 2:
            item["proposals"] = [
                _proposal_status(p) for p in proposals.parse_proposals(row["body"])
            ]
        summarized.append(item)
    return {"reports": summarized}


def _proposal_status(p: proposals.ParsedProposal) -> dict[str, Any]:
    try:
        selected, option = p.selection()
        state = "selected" if selected else "unselected"
    except proposals.SelectionError as exc:
        state, option = f"ambiguous: {exc}", None
    return {
        "id": p.pid,
        "title": p.title,
        "state": state,
        "option": option,
        "child_issue": p.child,
    }


def _list_since(
    args: list[str], *, key: str, since: datetime, limit: int, cwd: Path
) -> list[dict[str, Any]]:
    """List newest-first gh rows and keep those with ``key >= since``.

    Fails if the limit was hit while every returned row is still inside the period
    (i.e. older in-period rows may be missing).
    """
    rows = shell.gh_json([*args, "--limit", str(limit)], cwd=cwd)
    if not isinstance(rows, list):
        raise CommandError(f"gh {shlex.join(args)} did not return a JSON list")
    dated = [r for r in rows if r.get(key)]
    kept = [r for r in dated if parse_ts(r[key]) >= since]
    if len(rows) >= limit and len(kept) == len(dated):
        raise CommandError(
            f"gh {shlex.join(args)} hit --limit {limit} inside the report period"
        )
    return kept


def collect_github(cfg: CollectConfig, since: datetime) -> dict[str, Any]:
    cwd = cfg.repo_root
    open_prs = shell.gh_list(
        ["pr", "list", "--state", "open", "--json", _PR_FIELDS], limit=300, cwd=cwd
    )
    prs = []
    for pr in sorted(open_prs, key=lambda r: int(r["number"])):
        idle = _days(cfg.now, pr["updatedAt"])
        prs.append(
            {
                "number": pr["number"],
                "title": pr["title"],
                "branch": pr["headRefName"],
                "author": pr["author"]["login"],
                "draft": pr["isDraft"],
                "labels": _names(pr["labels"]),
                "created_at": pr["createdAt"],
                "updated_at": pr["updatedAt"],
                "idle_days": idle,
                "stale": idle >= cfg.stale_pr_days,
                "url": pr["url"],
            }
        )
    merged = _list_since(
        [
            "pr",
            "list",
            "--state",
            "merged",
            "--json",
            "number,title,headRefName,mergedAt",
        ],
        key="mergedAt",
        since=since,
        limit=300,
        cwd=cwd,
    )
    issues = shell.gh_list(
        ["issue", "list", "--state", "open", "--json", _ISSUE_FIELDS],
        limit=2000,
        cwd=cwd,
    )
    label_counts: Counter[str] = Counter()
    for issue in issues:
        label_counts.update(_names(issue["labels"]))
    ai_proposed = [
        {
            "number": i["number"],
            "title": i["title"],
            "age_days": _days(cfg.now, i["createdAt"]),
            "idle_days": _days(cfg.now, i["updatedAt"]),
        }
        for i in sorted(issues, key=lambda r: int(r["number"]))
        if AI_PROPOSED_LABEL in _names(i["labels"])
    ]
    oldest_idle = sorted(issues, key=lambda r: parse_ts(r["updatedAt"]))[:15]
    runs = _list_since(
        [
            "run", "list",
            "--json", "databaseId,workflowName,conclusion,status,headBranch,event,createdAt,url",
        ],
        key="createdAt",
        since=since,
        limit=1000,
        cwd=cwd,
    )  # fmt: skip
    conclusions = Counter(str(r.get("conclusion") or r.get("status")) for r in runs)
    failures = [r for r in runs if r.get("conclusion") == "failure"]
    return {
        "open_prs": prs,
        "stale_pr_days": cfg.stale_pr_days,
        "merged_prs_in_period": [
            {"number": p["number"], "title": p["title"], "branch": p["headRefName"]}
            for p in merged
        ],
        "open_issues": {
            "total": len(issues),
            "by_label": dict(sorted(label_counts.items())),
            "ai_proposed": ai_proposed,
            "least_recently_updated": [
                {
                    "number": i["number"],
                    "title": i["title"],
                    "idle_days": _days(cfg.now, i["updatedAt"]),
                }
                for i in oldest_idle
            ],
        },
        "ci_runs_in_period": {
            "total": len(runs),
            "by_conclusion": dict(sorted(conclusions.items())),
            "failures": [
                {
                    "workflow": r["workflowName"],
                    "branch": r["headBranch"],
                    "event": r["event"],
                    "created_at": r["createdAt"],
                    "url": r["url"],
                }
                for r in failures[:30]
            ],
        },
    }


def _last_commit_date(cfg: CollectConfig, path: Path) -> str:
    out = shell.git(
        ["log", "-1", "--format=%cs", cfg.base_ref, "--", str(path)], cwd=cfg.repo_root
    ).strip()
    return out or "uncommitted"


def collect_git(cfg: CollectConfig, since: datetime) -> dict[str, Any]:
    cwd = cfg.repo_root
    log = shell.git(
        [
            "log",
            cfg.base_ref,
            f"--since={since.isoformat()}",
            "--format=%h%x09%an%x09%s",
        ],
        cwd=cwd,
    ).splitlines()
    shortstat = shell.git(
        [
            "log",
            cfg.base_ref,
            f"--since={since.isoformat()}",
            "--shortstat",
            "--format=",
        ],
        cwd=cwd,
    )
    added = removed = 0
    for line in shortstat.splitlines():
        for part in line.split(","):
            words = part.split()
            if len(words) >= 2 and words[1].startswith("insertion"):
                added += int(words[0])
            elif len(words) >= 2 and words[1].startswith("deletion"):
                removed += int(words[0])
    line_counts = shell.git(
        ["grep", "-c", "", cfg.base_ref, "--", *CODE_SUFFIXES, *EXCLUDED_PATHSPECS],
        cwd=cwd,
        ok_codes=(0, 1),
    )
    files: list[tuple[str, int]] = []
    for row in line_counts.splitlines():
        path, _, count = row.removeprefix(f"{cfg.base_ref}:").rpartition(":")
        files.append((path, int(count)))
    files.sort(key=lambda item: (-item[1], item[0]))
    todo = shell.git(
        [
            "grep", "-I", "-c", "-E", r"\b(TODO|FIXME|XXX|HACK)\b", cfg.base_ref,
            "--", *EXCLUDED_PATHSPECS,
        ],
        cwd=cwd,
        ok_codes=(0, 1),
    )  # fmt: skip
    todo_files: list[tuple[str, int]] = []
    for row in todo.splitlines():
        path, _, count = row.removeprefix(f"{cfg.base_ref}:").rpartition(":")
        todo_files.append((path, int(count)))
    todo_files.sort(key=lambda item: (-item[1], item[0]))
    return {
        "commits_in_period": len(log),
        "commit_subjects": log[:120],
        "lines_added": added,
        "lines_removed": removed,
        "code_files_total": len(files),
        "code_lines_total": sum(c for _, c in files),
        "largest_code_files": [{"path": p, "lines": c} for p, c in files[:25]],
        "todo_markers_total": sum(c for _, c in todo_files),
        "todo_markers_top_files": [{"path": p, "count": c} for p, c in todo_files[:15]],
    }


def collect_workspace(cfg: CollectConfig) -> dict[str, Any]:
    cwd = cfg.repo_root
    porcelain = shell.git(["worktree", "list", "--porcelain"], cwd=cwd).splitlines()
    worktrees = [ln for ln in porcelain if ln.startswith("worktree ")]
    branches = shell.git(
        ["for-each-ref", "refs/heads", "--format=%(refname:short)"], cwd=cwd
    ).splitlines()
    merged = shell.git(
        ["branch", "--merged", cfg.base_ref, "--format=%(refname:short)"], cwd=cwd
    ).splitlines()
    return {
        "worktrees": len(worktrees),
        "prunable_worktrees": sum(1 for ln in porcelain if ln.startswith("prunable")),
        "local_branches": len(branches),
        "local_branches_merged_into_base": len(merged),
        "branches_by_prefix": dict(
            sorted(
                Counter(
                    b.split("/", 1)[0] if "/" in b else "(none)" for b in branches
                ).items()
            )
        ),
    }


def _cleanup_command(cfg: CollectConfig) -> tuple[str, ...]:
    if cfg.cleanup_cmd is not None:
        return cfg.cleanup_cmd
    entry = cfg.repo_root / CLEANUP_DIR / "cleanup.py"
    if not entry.is_file():
        raise CommandError(
            f"{CLEANUP_DIR} exists but has no cleanup.py; "
            "set WEEKLY_REPORT_CLEANUP_CMD to a command accepting --report-json FILE"
        )
    return (sys.executable, str(entry), "scan")


def collect_cleanup(cfg: CollectConfig) -> dict[str, Any]:
    if cfg.cleanup_cmd is None and not (cfg.repo_root / CLEANUP_DIR).is_dir():
        return {
            "status": "not_installed",
            "note": f"{CLEANUP_DIR} が存在しない（掃除モジュール未導入）",
        }
    # Contract (see .agents/ops/cleanup/README.md): the command writes its report to
    # the file given after --report-json and never deletes anything in scan mode.
    with tempfile.TemporaryDirectory(prefix="weekly-report-cleanup-") as tmp:
        out = Path(tmp) / "cleanup.json"
        cmd = (*_cleanup_command(cfg), "--report-json", str(out))
        shell.run(cmd, cwd=cfg.repo_root, timeout=900)
        if not out.is_file():
            raise CommandError(f"{shlex.join(cmd)} did not write {out}")
        try:
            parsed = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CommandError(f"{shlex.join(cmd)} wrote invalid JSON") from exc
    text = json.dumps(parsed, ensure_ascii=False, indent=1)
    shown = shlex.join((*cmd[:-1], "<tmp>"))
    return {"status": "ok", "command": shown, "report_json": _truncate(text)}


def collect_memory(cfg: CollectConfig) -> dict[str, Any]:
    mem_dir = cfg.repo_root / MEMORY_DIR
    if not mem_dir.is_dir():
        return {
            "status": "not_installed",
            "note": f"{MEMORY_DIR} が存在しない（共有memory未導入）",
        }
    index = mem_dir / "INDEX.md"
    entries = []
    for path in sorted(mem_dir.rglob("*.md")):
        if path == index:
            continue
        rel = path.relative_to(cfg.repo_root)
        head = path.read_text(encoding="utf-8").splitlines()[:15]
        entries.append(
            {"path": str(rel), "last_commit": _last_commit_date(cfg, rel), "head": head}
        )
    return {
        "status": "ok",
        "index": _truncate(index.read_text(encoding="utf-8"))
        if index.is_file()
        else None,
        "entries": entries[:300],
        "entries_total": len(entries),
    }


def collect_knowledge(cfg: CollectConfig, since: datetime) -> dict[str, Any]:
    summary = cfg.repo_root / KNOWLEDGE_SUMMARY
    if not summary.is_file():
        return {"status": "not_found", "note": f"{KNOWLEDGE_SUMMARY} が存在しない"}
    changed = shell.git(
        [
            "log", cfg.base_ref, f"--since={since.isoformat()}", "--name-only", "--format=",
            "--", "knowledge/nodes", "knowledge/reports",
        ],
        cwd=cfg.repo_root,
    )  # fmt: skip
    paths = sorted({ln for ln in changed.splitlines() if ln.strip()})
    return {
        "status": "ok",
        "summary_path": str(KNOWLEDGE_SUMMARY),
        "summary_last_commit": _last_commit_date(cfg, KNOWLEDGE_SUMMARY),
        "changed_in_period": paths[:60],
        "changed_in_period_total": len(paths),
    }


def collect_ai_env(cfg: CollectConfig) -> dict[str, Any]:
    tracked = shell.git(
        [
            "ls-tree", "-r", "--name-only", cfg.base_ref, "--",
            "AGENTS.md", "CLAUDE.md", ".agents", ".github/agents",
            ".github/copilot-instructions.md", ".claude",
        ],
        cwd=cfg.repo_root,
    ).splitlines()  # fmt: skip
    control = [
        p
        for p in tracked
        if p.endswith(".md")
        or "/systemd/" in p
        or p.endswith((".sh", ".toml", ".json"))
    ]
    return {
        "files": [
            {"path": p, "last_commit": _last_commit_date(cfg, Path(p))}
            for p in control[:150]
        ],
        "files_total": len(control),
    }


def collect_meta(cfg: CollectConfig, since: datetime) -> dict[str, Any]:
    cwd = cfg.repo_root
    base_sha = shell.git(["rev-parse", "--short", cfg.base_ref], cwd=cwd).strip()
    head_sha = shell.git(["rev-parse", "--short", "HEAD"], cwd=cwd).strip()
    branch = shell.git(["branch", "--show-current"], cwd=cwd).strip() or "(detached)"
    dirty = shell.git(["status", "--porcelain", "--untracked-files=no"], cwd=cwd)
    return {
        "generated_at": cfg.now.isoformat(timespec="minutes"),
        "period_start": since.isoformat(timespec="minutes"),
        "base_ref": cfg.base_ref,
        "base_sha": base_sha,
        "checkout_branch": branch,
        "checkout_sha": head_sha,
        "checkout_matches_base": base_sha == head_sha,
        "checkout_modified_files": len(dirty.splitlines()),
    }


def collect_all(cfg: CollectConfig) -> dict[str, Any]:
    history = collect_report_history(cfg)
    reports = history["reports"]
    since = (
        parse_ts(reports[-1]["createdAt"]).astimezone(cfg.now.tzinfo)
        if reports
        else cfg.now - timedelta(days=DEFAULT_PERIOD_DAYS)
    )
    return {
        "meta": collect_meta(cfg, since),
        "previous_reports": history,
        "github": collect_github(cfg, since),
        "git": collect_git(cfg, since),
        "workspace": collect_workspace(cfg),
        "cleanup": collect_cleanup(cfg),
        "memory": collect_memory(cfg),
        "knowledge": collect_knowledge(cfg, since),
        "ai_env": collect_ai_env(cfg),
    }
