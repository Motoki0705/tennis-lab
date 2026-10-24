"""Explicit sync and bounded local evidence reads; no agents or scheduling."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ci_watch.store import Store as ObservationStore
from issues import ci_failure_marker

from .github import GitHub
from .loop import sync
from .store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    state_root = (
        Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        / "tennis-lab-agents"
    )
    parser.add_argument("--state-dir", type=Path, default=state_root / "ci-failures")
    commands = parser.add_subparsers(dest="command", required=True)
    process = commands.add_parser("sync")
    process.add_argument("--repo", required=True)
    process.add_argument("--source-dir", required=True, type=Path)
    process.add_argument("--apply", action="store_true", required=True)
    records = commands.add_parser("records")
    records.add_argument("--after", type=int, default=0)
    records.add_argument("--limit", type=int, default=100)
    show = commands.add_parser("show")
    show.add_argument("--signature", required=True)
    summary = commands.add_parser("list")
    summary.add_argument("--offset", type=int, default=0)
    summary.add_argument("--limit", type=int, default=20)
    summary.add_argument("--unresolved", action="store_true")
    original = commands.add_parser("original")
    original.add_argument("--reference", required=True, type=Path)
    blob = commands.add_parser("evidence")
    blob.add_argument("--object", required=True)
    blob.add_argument("--start-line", type=int, default=1)
    blob.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    state = None
    source = None
    try:
        if args.command == "sync":
            gh = GitHub(args.repo)
            lock_path = (
                state_root
                / "ci-failure-locks"
                / (gh.repo.casefold().replace("/", "--") + ".lock")
            )
            os.umask(0o077)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(
                        "another CI failure sync holds the repository lock"
                    ) from None
                source = ObservationStore(args.source_dir.resolve(), readonly=True)
                state = Store(args.state_dir)
                result = sync(gh, source, state, datetime.now(UTC).isoformat())
            print(json.dumps(result, ensure_ascii=False))
            return 1 if result["errors"] else 0
        state = Store(args.state_dir, readonly=True)
        state.db.execute("BEGIN")
        if hasattr(args, "limit") and not 1 <= args.limit <= 1000:
            raise ValueError("--limit must be 1..1000")
        if args.command == "records":
            if args.after < 0:
                raise ValueError("--after must be nonnegative")
            result = state.records(args.after, args.limit)
        elif args.command == "show":
            ci_failure_marker(args.signature)
            result = state.show(args.signature)
        elif args.command == "list":
            if args.offset < 0:
                raise ValueError("--offset must be nonnegative")
            result = state.summaries(args.offset, args.limit, args.unresolved)
        elif args.command == "original":
            result = state.original(
                json.loads(args.reference.read_text(encoding="utf-8"))
            )
        else:
            if args.start_line < 1:
                raise ValueError("--start-line must be positive")
            lines = state.object(args.object).splitlines()
            start = args.start_line - 1
            result = {
                "object": args.object,
                "lines": lines[start : start + args.limit],
                "start_line": args.start_line,
                "total_lines": len(lines),
                "has_more": start + args.limit < len(lines),
            }
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        sqlite3.Error,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"ci_failures: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.close()
        if state is not None:
            state.db.close()
