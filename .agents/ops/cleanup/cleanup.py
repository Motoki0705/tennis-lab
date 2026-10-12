#!/usr/bin/env python3
"""Classify local branches and git worktrees and delete the safe ones (issue #1060).

Subcommands (see README.md for the policy and the report schema):

    cleanup.py scan        [--report-json FILE] [--no-disk]   # dry run, never deletes
    cleanup.py apply-auto  [--report-json FILE] [--no-disk]   # delete auto_delete branches
    cleanup.py delete ID [ID ...]                             # delete approved entries

``apply-auto`` only deletes branch refs that have no worktree; it never removes a
worktree. A worktree is removed only by ``delete worktree:<path>[@<sha>]`` and
only when it is completely clean (see README.md).

Common options: --repo DIR (default: main checkout of the cwd's repository),
--config FILE (default: config.toml next to this script), --state-dir DIR
(default: ${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/cleanup).

Exit status: 0 on success, 1 when a probe or a deletion failed or was refused,
2 on usage errors, 3 when another mutating cleanup run holds the lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from classify import BranchFacts, WorktreeFacts
from inventory import (
    DISPOSABLE_SYMLINK,
    CleanupError,
    Config,
    Entry,
    Inventory,
    build_inventory,
    git,
    iso,
    load_config,
    main_checkout,
)

SCHEMA_VERSION = 2
SCRIPT_DIR = Path(__file__).resolve().parent
EXIT_FAILURE, EXIT_USAGE, EXIT_LOCKED = 1, 2, 3


class LockedError(CleanupError):
    pass


class DriftError(CleanupError):
    """The entry changed since it was classified, or may not be deleted; nothing was removed."""


def default_state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    return Path(base) / "tennis-lab-agents" / "cleanup"


@contextmanager
def exclusive_lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "cleanup.lock", "w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockedError(
                f"another cleanup run holds {state_dir / 'cleanup.lock'}"
            ) from exc
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield


# ------------------------------------------------------------------------- report


def entry_json(entry: Entry) -> dict[str, Any]:
    state = entry.state
    return {
        "id": entry.id,
        "kind": "branch" if isinstance(entry.facts, BranchFacts) else "worktree",
        "category": entry.decision.category,
        "branch": entry.branch,
        "tip": entry.tip,
        "worktree": entry.worktree,
        "reasons": [
            {"code": r.code, "kind": r.kind, "detail": r.detail}
            for r in entry.decision.reasons
        ],
        "prs": [p.as_json() for p in entry.prs],
        "last_activity": iso(entry.last_activity),
        "activity_source": entry.activity_source,
        "idle_days": round(entry.facts.idle_hours / 24.0, 2),
        "worktree_state": None
        if state is None
        else {
            "dirty_tracked": list(state.dirty_tracked),
            "untracked": list(state.untracked),
            "ignored": list(state.ignored),
            "disposable": list(state.disposable),
            "submodule_populated": list(state.submodule_populated),
        },
        "disk_bytes": entry.disk.bytes if entry.disk else None,
        "disk_error": entry.disk.error if entry.disk else None,
        "action": entry.action,
        "action_error": entry.action_error,
    }


def build_report(inv: Inventory, cfg: Config, mode: str) -> dict[str, Any]:
    entries = sorted(inv.entries, key=lambda e: (e.decision.category, e.id))
    counts = Counter(e.decision.category for e in entries)
    reason_counts: dict[str, Counter[str]] = {}
    disk_by_category: Counter[str] = Counter()
    for e in entries:
        bucket = reason_counts.setdefault(e.decision.category, Counter())
        for code in {r.code for r in e.decision.reasons}:
            bucket[code] += 1
        if e.disk is not None and e.disk.bytes is not None:
            disk_by_category[e.decision.category] += e.disk.bytes
    sized = [e for e in entries if e.disk is not None]
    sized.sort(key=lambda e: -(e.disk.bytes or 0) if e.disk else 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": iso(time.time()),
        "host": socket.gethostname(),
        "repo": str(inv.repo),
        "mode": mode,
        "config": cfg.as_json(),
        "summary": {
            "branches": sum(1 for e in entries if isinstance(e.facts, BranchFacts)),
            "worktrees": sum(1 for e in entries if isinstance(e.facts, WorktreeFacts)),
            "categories": {
                c: counts.get(c, 0)
                for c in ("auto_delete", "approval_required", "protected")
            },
            "reason_counts": {
                c: dict(sorted(v.items())) for c, v in sorted(reason_counts.items())
            },
            "disk_bytes_by_category": dict(sorted(disk_by_category.items())),
            "disk_bytes_total": sum(disk_by_category.values()),
            "actions": dict(Counter(e.action for e in entries if e.action != "none")),
        },
        "worktrees_by_size": [
            {
                "worktree": e.worktree,
                "id": e.id,
                "category": e.decision.category,
                "disk_bytes": e.disk.bytes if e.disk else None,
                "disk_error": e.disk.error if e.disk else None,
            }
            for e in sized
        ],
        "warnings": inv.warnings,
        "entries": [entry_json(e) for e in entries],
    }


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f}G"


def print_summary(report: dict[str, Any], out: Any = sys.stdout) -> None:
    s = report["summary"]
    print(f"repo: {report['repo']}  mode: {report['mode']}", file=out)
    print(f"branches: {s['branches']}  worktrees: {s['worktrees']}", file=out)
    for cat, n in s["categories"].items():
        reasons = s["reason_counts"].get(cat, {})
        top = ", ".join(
            f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])
        )
        disk = s["disk_bytes_by_category"].get(cat, 0)
        print(f"  {cat:18s} {n:4d}  disk={_gib(disk):>7s}  [{top}]", file=out)
    if s["actions"]:
        print(f"actions: {s['actions']}", file=out)
    for w in report["warnings"]:
        print(f"warning: {w}", file=out)
    for e in report["entries"]:
        if e["category"] == "auto_delete" or e["action"] != "none":
            print(f"  {e['action']:12s} {e['category']:18s} {e['id']}", file=out)


# ----------------------------------------------------------------------- deletion


def append_log(state_dir: Path, record: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": iso(time.time()), **record}, ensure_ascii=False)
    with open(state_dir / "deleted.jsonl", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _delete_branch(repo: Path, entry: Entry, f: BranchFacts) -> None:
    ref = f"refs/heads/{f.branch}"
    symbolic = git(repo, "symbolic-ref", "-q", ref, ok_codes=(0, 1)).stdout.strip()
    if symbolic:
        raise DriftError(f"{entry.id}: {ref} is now a symbolic ref to {symbolic}")
    now = git(repo, "rev-parse", "--verify", "-q", ref, ok_codes=(0, 1)).stdout.strip()
    if now != f.tip:
        raise DriftError(
            f"{entry.id}: branch tip is {now or '(deleted)'}, classified at {f.tip}"
        )
    # Compare-and-delete (fails if the tip moved); --no-deref never follows a
    # symbolic ref to its target (e.g. main).
    git(repo, "update-ref", "--no-deref", "-d", ref, f.tip)
    section = f"branch.{f.branch}"
    if git(
        repo,
        "config",
        "--get-regexp",
        "^" + re.escape(section) + r"\.",
        ok_codes=(0, 1),
    ).stdout:
        git(repo, "config", "--remove-section", section)


def _remove_worktree(repo: Path, entry: Entry, f: WorktreeFacts) -> None:
    wt = Path(f.worktree)
    head = git(wt, "rev-parse", "HEAD").stdout.strip()
    if head != f.head:
        raise DriftError(f"{entry.id}: worktree HEAD is {head}, classified at {f.head}")
    venv = wt / DISPOSABLE_SYMLINK
    target: str | None = None
    if f.state is not None and f.state.disposable:
        if not venv.is_symlink():
            raise DriftError(f"{entry.id}: {venv} is no longer a symlink")
        # `git worktree remove` counts the symlink as untracked; unlink only it
        # (never its target) so git still re-checks everything else.
        target = os.readlink(venv)
        venv.unlink()
    try:
        git(repo, "worktree", "remove", f.worktree)  # never --force
    except CleanupError:
        if target is not None and not venv.is_symlink():
            venv.symlink_to(target)
        raise


def delete_entry(repo: Path, entry: Entry, state_dir: Path, *, approved: bool) -> None:
    """Delete one branch ref or remove one worktree, logging before and after.

    A worktree removal never deletes its branch, and a branch is only deleted
    when no worktree checks it out.
    """
    f = entry.facts
    base = {
        "id": entry.id,
        "repo": str(repo),
        "branch": entry.branch,
        "tip": entry.tip,
        "worktree": entry.worktree,
        "category": entry.decision.category,
        "approved": approved,
        "reasons": [
            {"code": r.code, "detail": r.detail} for r in entry.decision.reasons
        ],
    }
    append_log(state_dir, {**base, "phase": "intent"})
    try:
        if isinstance(f, BranchFacts):
            _delete_branch(repo, entry, f)
        else:
            _remove_worktree(repo, entry, f)
    except CleanupError as exc:
        append_log(state_dir, {**base, "phase": "failed", "error": str(exc)})
        raise
    append_log(state_dir, {**base, "phase": "done"})


def reverify(repo: Path, cfg: Config, inv: Inventory, entry: Entry, cwd: Path) -> Entry:
    """Re-collect live facts (tip, worktree status, processes, queue) right before deleting."""
    fresh = build_inventory(
        repo,
        cfg,
        invoking_cwd=cwd,
        measure_disk=False,
        only_ids={entry.id},
        prs=inv.prs,
    )
    if not fresh.entries:
        raise DriftError(f"{entry.id}: no longer exists")
    new = fresh.entries[0]
    if new.tip != entry.tip or new.branch != entry.branch:
        raise DriftError(
            f"{entry.id}: tip or branch changed since classification "
            f"({entry.tip} -> {new.tip}, {entry.branch} -> {new.branch})"
        )
    new.disk = entry.disk
    return new


def describe_drift(old: Entry, new: Entry) -> str | None:
    """What changed in the verdict or worktree contents between two snapshots."""
    changes: list[str] = []
    if old.decision.category != new.decision.category:
        changes.append(f"category {old.decision.category} -> {new.decision.category}")
    before = {r.code for r in old.decision.reasons if r.kind != "info"}
    after = {r.code for r in new.decision.reasons if r.kind != "info"}
    if before != after:
        changes.append(
            "reasons "
            + " ".join(
                [f"+{c}" for c in sorted(after - before)]
                + [f"-{c}" for c in sorted(before - after)]
            )
        )
    if old.state != new.state:
        changes.append("worktree contents changed")
    return "; ".join(changes) or None


def refusal(entry: Entry) -> str | None:
    """Why an approved ``delete`` must not touch this entry, or None."""
    for kind, label in (("protect", "protected"), ("block", "blocked")):
        found = [r for r in entry.decision.reasons if r.kind == kind]
        if found:
            return f"{label}: " + "; ".join(f"{r.code}: {r.detail}" for r in found)
    return None


# ---------------------------------------------------------------------------- CLI


def parse_id(raw: str) -> tuple[str, str | None]:
    """``branch:<name>[@<sha>]``, ``worktree:<path>[@<sha>]`` or a bare branch name."""
    entry_id = (
        raw if raw.startswith("worktree:") else "branch:" + raw.removeprefix("branch:")
    )
    m = re.fullmatch(r"(.+)@([0-9a-f]{7,40})", entry_id)
    if m:
        return m.group(1), m.group(2)
    return entry_id, None


def cmd_scan(args: argparse.Namespace, repo: Path, cfg: Config) -> int:
    inv = build_inventory(
        repo, cfg, invoking_cwd=Path.cwd(), measure_disk=not args.no_disk
    )
    for e in inv.entries:
        if e.decision.category == "auto_delete":
            e.action = "would_delete"
    return finish(args, inv, cfg, "dry-run", failed=False)


def cmd_apply_auto(args: argparse.Namespace, repo: Path, cfg: Config) -> int:
    with exclusive_lock(args.state_dir):
        inv = build_inventory(
            repo, cfg, invoking_cwd=Path.cwd(), measure_disk=not args.no_disk
        )
        failed = False
        for i, e in enumerate(inv.entries):
            if e.decision.category != "auto_delete":
                continue
            if not isinstance(e.facts, BranchFacts):
                raise CleanupError(f"{e.id}: only branches can be auto_delete")
            try:
                live = reverify(repo, cfg, inv, e, Path.cwd())
                inv.entries[i] = live
                if live.decision.category != "auto_delete":
                    live.action = "skipped"
                    live.action_error = (
                        f"reclassified as {live.decision.category} on re-check"
                    )
                    continue
                delete_entry(repo, live, args.state_dir, approved=False)
                live.action = "deleted"
            except CleanupError as exc:
                inv.entries[i].action = "failed"
                inv.entries[i].action_error = str(exc)
                failed = True
        return finish(args, inv, cfg, "apply-auto", failed=failed)


def cmd_delete(args: argparse.Namespace, repo: Path, cfg: Config) -> int:
    wanted = [parse_id(raw) for raw in args.ids]
    with exclusive_lock(args.state_dir):
        inv = build_inventory(
            repo,
            cfg,
            invoking_cwd=Path.cwd(),
            measure_disk=False,
            only_ids={i for i, _ in wanted},
        )
        index = {e.id: i for i, e in enumerate(inv.entries)}
        failed = False
        for entry_id, expect in wanted:
            if entry_id not in index:
                print(f"error: {entry_id}: no such branch/worktree", file=sys.stderr)
                failed = True
                continue
            i = index[entry_id]
            snapshot = inv.entries[i]
            try:
                # Earlier deletions in this run take time; re-collect tip, worktree
                # status, processes and queue jobs for this entry now.
                entry = reverify(repo, cfg, inv, snapshot, Path.cwd())
                inv.entries[i] = entry
                problem = describe_drift(snapshot, entry)
                if problem is not None:
                    problem = f"changed since classification: {problem}"
                else:
                    problem = refusal(entry)
                if (
                    problem is None
                    and expect is not None
                    and not entry.tip.startswith(expect)
                ):
                    problem = f"tip is {entry.tip}, approval was for {expect}"
                if problem is not None:
                    raise DriftError(problem)
                delete_entry(repo, entry, args.state_dir, approved=True)
                entry.action = "deleted"
            except CleanupError as exc:
                current = inv.entries[i]
                current.action = "refused" if isinstance(exc, DriftError) else "failed"
                current.action_error = str(exc)
                failed = True
        return finish(args, inv, cfg, "delete", failed=failed)


def finish(
    args: argparse.Namespace, inv: Inventory, cfg: Config, mode: str, *, failed: bool
) -> int:
    report = build_report(inv, cfg, mode)
    if args.report_json is not None:
        write_json_atomic(args.report_json, report)
    print_summary(report)
    for e in inv.entries:
        if e.action_error:
            label = "note" if e.action == "skipped" else "error"
            print(f"{label}: {e.id}: {e.action_error}", file=sys.stderr)
    return EXIT_FAILURE if failed else 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repo",
        type=Path,
        help="repository (default: main checkout of the cwd's repo)",
    )
    common.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.toml")
    common.add_argument("--state-dir", type=Path, default=default_state_dir())
    common.add_argument(
        "--report-json", type=Path, help="write the report JSON (schema: README.md)"
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "apply-auto"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--no-disk", action="store_true", help="skip per-worktree du")
    p = sub.add_parser("delete", parents=[common])
    p.add_argument(
        "ids",
        nargs="+",
        help="branch:<name>[@<sha>], worktree:<path>[@<sha>], or a bare branch name",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        repo = (args.repo or main_checkout(Path.cwd())).resolve()
        if main_checkout(repo).resolve() != repo:
            raise CleanupError(f"--repo must be the main checkout, got {repo}")
        handler = {
            "scan": cmd_scan,
            "apply-auto": cmd_apply_auto,
            "delete": cmd_delete,
        }[args.command]
        return handler(args, repo, cfg)
    except LockedError as exc:
        print(f"cleanup: {exc}", file=sys.stderr)
        return EXIT_LOCKED
    except CleanupError as exc:
        print(f"cleanup: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
