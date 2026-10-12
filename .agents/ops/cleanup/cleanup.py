#!/usr/bin/env python3
"""Classify and clean up local branches and git worktrees (issue #1060).

Subcommands (see README.md for the policy and the report schema):

    cleanup.py scan        [--report-json FILE] [--no-disk]   # dry run, never deletes
    cleanup.py apply-auto  [--report-json FILE] [--no-disk]   # delete auto_delete entries only
    cleanup.py delete ID [ID ...] [--discard-changes]         # delete approved entries

Common options: --repo DIR (default: main checkout of the cwd's repository),
--config FILE (default: config.toml next to this script), --state-dir DIR
(default: ${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/cleanup).

Exit status: 0 on success, 1 when a probe or a deletion failed, 2 on usage
errors, 3 when another mutating cleanup run holds the lock.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from inventory import (
    CleanupError,
    Config,
    Entry,
    Inventory,
    build_inventory,
    discard_fingerprint,
    git,
    iso,
    load_config,
    main_checkout,
)

SCHEMA_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent
EXIT_FAILURE, EXIT_USAGE, EXIT_LOCKED = 1, 2, 3


class LockedError(CleanupError):
    pass


class DriftError(CleanupError):
    """The entry changed between classification and deletion; nothing was removed."""


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
    f = entry.facts
    state = f.worktree_state
    return {
        "id": entry.id,
        "category": entry.decision.category,
        "branch": f.branch,
        "tip": f.tip,
        "worktree": f.worktree,
        "reasons": [
            {"code": r.code, "kind": r.kind, "detail": r.detail}
            for r in entry.decision.reasons
        ],
        "prs": [p.as_json() for p in entry.prs],
        "last_activity": iso(entry.last_activity),
        "activity_source": entry.activity_source,
        "idle_days": round(f.idle_hours / 24.0, 2),
        "worktree_state": None
        if state is None
        else {
            "dirty_tracked": list(state.dirty_tracked),
            "untracked": list(state.untracked),
            "ignored": list(state.ignored),
            "disposable_untracked": list(state.disposable_untracked),
            "disposable_ignored": list(state.disposable_ignored),
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
            "branches": sum(1 for e in entries if e.facts.branch is not None),
            "worktrees": sum(1 for e in entries if e.facts.worktree is not None),
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
                "worktree": e.facts.worktree,
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


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:80]


def _remove_disposable(worktree: Path, rels: Sequence[str]) -> None:
    root = worktree.resolve()
    for rel in rels:
        path = worktree / rel.rstrip("/")
        if path.parent.resolve() != root and root not in path.parent.resolve().parents:
            raise CleanupError(f"refusing to remove {path}: outside {root}")
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            raise CleanupError(
                f"disposable entry vanished or has an unknown type: {path}"
            )


def backup_name(entry_id: str) -> str:
    """Unique per entry and per call: the slug is lossy, so the id hash disambiguates."""
    digest = hashlib.sha256(entry_id.encode("utf-8")).hexdigest()[:16]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{time.time_ns()}-{_slug(entry_id)}-{digest}.patch"


def write_backup(state_dir: Path, entry_id: str, worktree: Path) -> Path:
    """Save ``git diff --binary HEAD`` of ``worktree``; never overwrites a file."""
    proc = subprocess.run(
        ["git", "--no-optional-locks", "diff", "--binary", "HEAD"],
        cwd=worktree,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CleanupError(
            f"git diff in {worktree} exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    backup = state_dir / "backups"
    backup.mkdir(parents=True, exist_ok=True)
    target = backup / backup_name(entry_id)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CleanupError(
            f"backup {target} already exists; refusing to overwrite"
        ) from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(proc.stdout)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _assert_unmoved(repo: Path, entry: Entry) -> None:
    """Compare tips before anything is removed (update-ref -d re-checks atomically)."""
    f = entry.facts
    if f.branch is not None:
        ref = f"refs/heads/{f.branch}"
        symbolic = git(repo, "symbolic-ref", "-q", ref, ok_codes=(0, 1)).stdout.strip()
        if symbolic:
            raise DriftError(f"{entry.id}: {ref} is now a symbolic ref to {symbolic}")
        now = git(
            repo,
            "rev-parse",
            "--verify",
            "-q",
            f"refs/heads/{f.branch}",
            ok_codes=(0, 1),
        ).stdout.strip()
        if now != f.tip:
            raise DriftError(
                f"{entry.id}: branch tip is {now or '(deleted)'}, classified at {f.tip}"
            )
    if f.worktree is not None and not f.worktree_missing:
        head = git(Path(f.worktree), "rev-parse", "HEAD").stdout.strip()
        if head != f.tip:
            raise DriftError(
                f"{entry.id}: worktree HEAD is {head}, classified at {f.tip}"
            )


def delete_entry(
    repo: Path, entry: Entry, state_dir: Path, *, approved: bool, discard: bool
) -> None:
    """Remove the worktree (if any) and the branch ref, logging before and after."""
    f = entry.facts
    state = f.worktree_state
    has_changes = state is not None and bool(
        state.dirty_tracked or state.untracked or state.ignored
    )
    if has_changes and not discard:
        raise CleanupError(
            f"{entry.id}: worktree has uncommitted/untracked/ignored files; pass --discard-changes"
        )
    patch_path: str | None = None
    if f.worktree is not None and state is not None and state.dirty_tracked:
        patch_path = str(write_backup(state_dir, entry.id, Path(f.worktree)))
    base = {
        "id": entry.id,
        "repo": str(repo),
        "branch": f.branch,
        "tip": f.tip,
        "worktree": f.worktree,
        "category": entry.decision.category,
        "approved": approved,
        "reasons": [
            {"code": r.code, "detail": r.detail} for r in entry.decision.reasons
        ],
        "discarded_patch": patch_path,
        "discarded_untracked": list(state.untracked + state.ignored)
        if state is not None and discard
        else [],
    }
    append_log(state_dir, {**base, "phase": "intent"})
    try:
        _assert_unmoved(repo, entry)
        if f.worktree is not None:
            force = approved and (
                has_changes or bool(state and state.submodule_populated)
            )
            if not force and state is not None:
                # `git worktree remove` refuses untracked entries even when they are
                # disposable symlinks; drop exactly those so git still re-checks
                # that nothing else is untracked or modified.
                _remove_disposable(Path(f.worktree), state.disposable_untracked)
            git(repo, "worktree", "remove", *(["--force"] if force else []), f.worktree)
        if f.branch is not None:
            # Compare-and-delete: fails if the tip moved since classification.
            # --no-deref: never follow a symbolic ref to its target (e.g. main).
            git(
                repo,
                "update-ref",
                "--no-deref",
                "-d",
                f"refs/heads/{f.branch}",
                f.tip or "",
            )
            section = f"branch.{f.branch}"
            if git(
                repo,
                "config",
                "--get-regexp",
                "^" + re.escape(section) + r"\.",
                ok_codes=(0, 1),
            ).stdout:
                git(repo, "config", "--remove-section", section)
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
    if new.facts.tip != entry.facts.tip or new.facts.worktree != entry.facts.worktree:
        raise DriftError(
            f"{entry.id}: tip or worktree changed since classification "
            f"({entry.facts.tip} -> {new.facts.tip}, {entry.facts.worktree} -> {new.facts.worktree})"
        )
    new.disk = entry.disk
    return new


def fingerprint(entry: Entry) -> dict[str, str]:
    f = entry.facts
    if f.worktree is None or f.worktree_state is None:
        return {}
    return discard_fingerprint(Path(f.worktree), f.worktree_state)


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
    if old.facts.worktree_state != new.facts.worktree_state:
        changes.append("worktree contents changed")
    return "; ".join(changes) or None


# ---------------------------------------------------------------------------- CLI


def parse_id(raw: str) -> tuple[str, str | None]:
    """``branch:<name>[@<sha>]``, ``worktree:<path>`` or a bare branch name."""
    if raw.startswith("worktree:"):
        return raw, None
    name = raw.removeprefix("branch:")
    expect: str | None = None
    m = re.fullmatch(r"(.+)@([0-9a-f]{7,40})", name)
    if m:
        name, expect = m.group(1), m.group(2)
    return f"branch:{name}", expect


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
            try:
                live = reverify(repo, cfg, inv, e, Path.cwd())
                inv.entries[i] = live
                if live.decision.category != "auto_delete":
                    live.action = "skipped"
                    live.action_error = (
                        f"reclassified as {live.decision.category} on re-check"
                    )
                    continue
                delete_entry(repo, live, args.state_dir, approved=False, discard=False)
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
        # Content of the files a --discard-changes deletion would destroy, taken
        # before anything is removed; re-taken right before each deletion.
        contents = {e.id: fingerprint(e) for e in inv.entries}
        failed = False
        for entry_id, expect in wanted:
            if entry_id not in index:
                print(f"error: {entry_id}: no such branch/worktree", file=sys.stderr)
                failed = True
                continue
            i = index[entry_id]
            snapshot = inv.entries[i]
            try:
                # Earlier deletions in this run can take minutes; re-collect tip,
                # worktree status, processes and queue jobs for this entry now.
                entry = reverify(repo, cfg, inv, snapshot, Path.cwd())
                inv.entries[i] = entry
                drift = describe_drift(snapshot, entry)
                if drift is None:
                    now = fingerprint(entry)
                    changed = sorted(
                        k
                        for k in contents[entry_id].keys() | now.keys()
                        if contents[entry_id].get(k) != now.get(k)
                    )
                    if changed:
                        drift = (
                            "local files changed: "
                            + ", ".join(changed[:5])
                            + (
                                f" (+{len(changed) - 5} more)"
                                if len(changed) > 5
                                else ""
                            )
                        )
                if drift is not None:
                    raise DriftError(f"changed since classification: {drift}")
            except CleanupError as exc:
                # Report on the newest snapshot (the re-collected one when it exists).
                current = inv.entries[i]
                current.action = "refused" if isinstance(exc, DriftError) else "failed"
                current.action_error = str(exc)
                failed = True
                continue
            if entry.decision.category == "protected":
                entry.action = "refused"
                entry.action_error = "protected: " + "; ".join(
                    f"{r.code}: {r.detail}"
                    for r in entry.decision.reasons
                    if r.kind == "protect"
                )
                failed = True
                continue
            if expect is not None and not (entry.facts.tip or "").startswith(expect):
                entry.action = "refused"
                entry.action_error = (
                    f"tip is {entry.facts.tip}, approval was for {expect}"
                )
                failed = True
                continue
            try:
                delete_entry(
                    repo,
                    entry,
                    args.state_dir,
                    approved=True,
                    discard=args.discard_changes,
                )
                entry.action = "deleted"
            except DriftError as exc:
                entry.action, entry.action_error = "refused", str(exc)
                failed = True
            except CleanupError as exc:
                entry.action = "failed"
                entry.action_error = str(exc)
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
        help="branch:<name>[@<sha>], worktree:<path>, or a bare branch name",
    )
    p.add_argument(
        "--discard-changes",
        action="store_true",
        help="allow removing worktrees with local changes",
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
