#!/usr/bin/env python3
"""
Read-only preflight inspection for PR creation in Motoki0705/tennis-lab.

Collects branch, commit, remote, existing-PR, label and PR-body facts, then
reports them as structured JSON so the caller can decide how to proceed.
This script never mutates the repository (the only exception is `--fetch`,
which updates remote-tracking refs).

Findings are reported, not raised: the script exits 0 whenever inspection
succeeded, even when blockers were found. It exits 2 only when inspection
itself is impossible (not a git repo, `gh` missing).

Usage:
    python3 .agents/skills/gh-pr-create/scripts/pr_preflight.py \
        [--base main] [--branch <branch>] [--body-file <path>] \
        [--issue <n>] [--label <name>]... [--fetch] [--format json|text]

Notes:
    - `blockers` must be resolved before creating a PR; `warnings` need a judgment call.
    - `suggested_next` is one of: create, update_existing, blocked.
    - Commit contents (base...HEAD), not the index, are inspected for stray paths,
      because those commits are what a push would publish.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = "Motoki0705/tennis-lab"
DEFAULT_BASE = "main"

# Paths that must never reach a PR. In a linked worktree, `.training_queue`
# and `.venv` are symlinks to the main checkout and are not gitignored there.
# `runs/` and `ckpt/` are anchored to the repo root: `knowledge/runs/` holds
# knowledge-control artifacts and is committed on purpose.
STRAY_PATH_PATTERNS = (
    re.compile(r"(^|/)\.venv(/|$)"),
    re.compile(r"(^|/)\.training_queue(/|$)"),
    re.compile(r"(^|/)__pycache__(/|$)"),
    re.compile(r"^runs/"),
    re.compile(r"^ckpt/"),
    re.compile(r"\.(ckpt|pt|pth|pkl)$"),
    re.compile(r"(^|/)\.env$"),
)

REQUIRED_BODY_HEADINGS = ("## 概要", "## 変更内容", "## 検証", "## 関連Issue")
TIMEOUT = 30


@dataclass
class Report:
    facts: dict[str, Any] = field(default_factory=dict)
    blockers: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)

    def block(self, code: str, message: str) -> None:
        self.blockers.append({"code": code, "message": message})

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})


def run(*cmd: str) -> tuple[int, str]:
    """Run a command, returning (returncode, stripped stdout)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return 124, ""
    except FileNotFoundError:
        return 127, ""
    return proc.returncode, proc.stdout.strip()


def die(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(2)


def lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def stray_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if any(pat.search(p) for pat in STRAY_PATH_PATTERNS)]


def inspect_git(rep: Report, base: str, branch: str | None, do_fetch: bool) -> str:
    rc, _ = run("git", "rev-parse", "--is-inside-work-tree")
    if rc != 0:
        die("not inside a git work tree")

    _, git_dir = run("git", "rev-parse", "--absolute-git-dir")
    _, common_dir = run("git", "rev-parse", "--path-format=absolute", "--git-common-dir")
    rep.facts["worktree"] = {
        "is_linked": git_dir != common_dir,
        "root": run("git", "rev-parse", "--show-toplevel")[1],
    }

    rc, head = run("git", "symbolic-ref", "--quiet", "--short", "HEAD")
    if rc != 0:
        rep.block("detached_head", "HEAD is detached; check out a branch first")
        head = ""
    branch = branch or head
    _, sha = run("git", "rev-parse", "--short", "HEAD")
    rep.facts["branch"] = branch
    rep.facts["head_sha"] = sha
    rep.facts["base"] = base

    if branch == base:
        rep.block("on_base_branch", f"current branch is the base branch '{base}'; create a topic branch")

    if do_fetch:
        run("git", "fetch", "--quiet", "origin", base)

    rc, _ = run("git", "rev-parse", "--verify", "--quiet", f"origin/{base}")
    if rc != 0:
        rep.block("missing_base_ref", f"origin/{base} not found locally; run with --fetch")
        return branch

    _, age = run("git", "log", "-1", "--format=%cr", f"origin/{base}")
    rep.facts["base_ref_age"] = age
    if not do_fetch:
        rep.warn("base_ref_not_refreshed", f"origin/{base} was last updated {age}; re-run with --fetch to be sure")

    _, ahead = run("git", "rev-list", "--count", f"origin/{base}..HEAD")
    _, behind = run("git", "rev-list", "--count", f"HEAD..origin/{base}")
    _, subjects = run("git", "log", "--format=%s", f"origin/{base}..HEAD")
    commit_subjects = lines(subjects)
    rep.facts["commits"] = {
        "ahead": int(ahead or 0),
        "behind": int(behind or 0),
        "subjects": commit_subjects[:20],
    }
    if int(ahead or 0) == 0:
        rep.block("no_commits", f"no commits between origin/{base} and HEAD; nothing to open a PR for")
    if int(behind or 0) > 0:
        rep.warn("base_advanced", f"base is {behind} commits ahead of HEAD; consider rebasing")

    _, changed = run("git", "diff", "--name-only", f"origin/{base}...HEAD")
    changed_files = lines(changed)
    rep.facts["changed_files"] = {"count": len(changed_files), "sample": changed_files[:20]}
    for path in stray_paths(changed_files):
        rep.block("stray_path_committed", f"commit contains a path that must not be pushed: {path}")

    _, raw = run("git", "diff", "--raw", f"origin/{base}...HEAD")
    for ln in lines(raw):
        # :000000 120000 0000000 abc1234 A\tpath  -> dst mode is field 1
        fields = ln.split()
        if len(fields) >= 2 and fields[1] == "120000":
            rep.block("symlink_committed", f"commit adds a symlink: {ln.split(chr(9))[-1]}")

    staged = lines(run("git", "diff", "--cached", "--name-only")[1])
    unstaged = lines(run("git", "diff", "--name-only")[1])
    untracked = lines(run("git", "ls-files", "--others", "--exclude-standard")[1])
    rep.facts["uncommitted"] = {
        "staged": len(staged),
        "unstaged": len(unstaged),
        "untracked": len(untracked),
    }
    if staged:
        rep.warn("staged_not_committed", f"{len(staged)} staged file(s) will NOT be in the PR; commit them first")
    if unstaged:
        rep.warn("unstaged_changes", f"{len(unstaged)} modified file(s) are not committed")

    rc, remote_head = run("git", "ls-remote", "--heads", "origin", branch)
    rep.facts["remote_branch_exists"] = bool(remote_head) if rc == 0 else None
    if rc != 0:
        rep.warn("ls_remote_failed", "could not reach origin to check whether the branch is pushed")
    return branch


def inspect_gh(rep: Report, branch: str, labels: list[str]) -> None:
    rc, _ = run("gh", "--version")
    if rc == 127:
        die("`gh` is not installed")

    rc, _ = run("gh", "auth", "status")
    authed = rc == 0
    rep.facts["gh_authenticated"] = authed
    if not authed:
        rep.block("gh_not_authenticated", "`gh auth status` failed; re-authenticate before creating a PR")
        return

    rc, out = run(
        "gh", "pr", "list", "--repo", REPO, "--head", branch,
        "--state", "all", "--limit", "10",
        "--json", "number,url,state,isDraft,baseRefName",
    )
    prs = json.loads(out) if rc == 0 and out else []
    open_prs = [p for p in prs if p.get("state") == "OPEN"]
    rep.facts["existing_prs"] = prs
    if open_prs:
        pr = open_prs[0]
        rep.warn("open_pr_exists", f"PR #{pr['number']} is already open for this branch: {pr['url']}")
    elif prs:
        closed = ", ".join(f"#{p['number']} ({p['state']})" for p in prs)
        rep.warn("closed_pr_exists", f"this branch previously had: {closed}")

    rc, out = run("gh", "label", "list", "--repo", REPO, "--limit", "100", "--json", "name")
    known = {item["name"] for item in json.loads(out)} if rc == 0 and out else set()
    rep.facts["known_labels"] = sorted(known)
    for label in labels:
        if known and label not in known:
            rep.block("unknown_label", f"label '{label}' does not exist in {REPO}")


def inspect_body(rep: Report, body_file: Path, issue: int | None) -> None:
    if not body_file.is_file():
        rep.block("body_missing", f"body file not found: {body_file}")
        return
    text = body_file.read_text(encoding="utf-8")
    if not text.strip():
        rep.block("body_empty", f"body file is empty: {body_file}")
        return

    problems: list[str] = []
    if "<!--" in text:
        problems.append("template HTML comments are still present")
    for ln in text.splitlines():
        stripped = ln.strip()
        if stripped == "-":
            problems.append("an unfilled placeholder bullet ('-') remains")
            break
    for ln in text.splitlines():
        if re.fullmatch(r"-\s*(Closes|References)\s*#\s*", ln.strip()):
            problems.append(f"an unfilled issue reference remains: {ln.strip()!r}")
            break

    missing = [h for h in REQUIRED_BODY_HEADINGS if h not in text]
    if missing:
        problems.append(f"missing template heading(s): {', '.join(missing)}")

    if issue is not None and not re.search(rf"(Closes|References)\s+#{issue}\b", text):
        problems.append(f"--issue {issue} given but body has no 'Closes #{issue}' / 'References #{issue}'")

    rep.facts["body"] = {"path": str(body_file), "bytes": len(text.encode()), "problems": problems}
    for problem in problems:
        rep.block("body_not_ready", problem)


def render_text(payload: dict[str, Any]) -> str:
    out = [f"suggested_next: {payload['suggested_next']}"]
    for kind in ("blockers", "warnings"):
        for item in payload[kind]:
            out.append(f"[{kind[:-1].upper()}] {item['code']}: {item['message']}")
    facts = payload["facts"]
    out.append(f"branch={facts.get('branch')} base={facts.get('base')} head={facts.get('head_sha')}")
    commits = facts.get("commits", {})
    out.append(f"commits ahead={commits.get('ahead')} behind={commits.get('behind')}")
    out.append(f"changed_files={facts.get('changed_files', {}).get('count')}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only PR preflight inspection.")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--branch", default=None, help="defaults to the current branch")
    parser.add_argument("--body-file", type=Path, default=None)
    parser.add_argument("--issue", type=int, default=None)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--fetch", action="store_true", help="refresh origin/<base> first")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args()

    rep = Report()
    branch = inspect_git(rep, args.base, args.branch, args.fetch)
    if branch:
        inspect_gh(rep, branch, args.label)
    if args.body_file is not None:
        inspect_body(rep, args.body_file, args.issue)

    has_open_pr = any(w["code"] == "open_pr_exists" for w in rep.warnings)
    if rep.blockers:
        suggested = "blocked"
    elif has_open_pr:
        suggested = "update_existing"
    else:
        suggested = "create"

    payload = {
        "suggested_next": suggested,
        "blockers": rep.blockers,
        "warnings": rep.warnings,
        "facts": rep.facts,
    }
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
