"""Manual collection and bounded local reads; no scheduler or GitHub writes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from .collect import collect
from .github import GitHub, sha
from .store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        / "tennis-lab-agents/ci-watch",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect")
    collection.add_argument("--repo", required=True)
    collection.add_argument("--start-sha")
    collection.add_argument(
        "--workflow", action="append", help="filename, repeatable; default ci.yml"
    )
    records = commands.add_parser("records")
    records.add_argument("--after", type=int, default=0)
    records.add_argument("--limit", type=int, default=100)
    records.add_argument("--sha")
    show = commands.add_parser("show")
    show.add_argument("--sha", required=True)
    evidence = commands.add_parser("evidence")
    evidence.add_argument("--object", required=True)
    evidence.add_argument("--start-line", type=int, default=1)
    evidence.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    store = None
    try:
        if args.command == "collect":
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
                raise ValueError("--repo must be owner/repository")
            workflows = sorted(set(args.workflow or ["ci.yml"]))
            if any(not re.fullmatch(r"[A-Za-z0-9_.-]+\.ya?ml", w) for w in workflows):
                raise ValueError("--workflow must be a workflow filename")
            config = {
                "schema_version": 1,
                "repository": args.repo,
                "branch": "main",
                "workflows": workflows,
                "start_sha": sha(args.start_sha) if args.start_sha else None,
            }
            os.umask(0o077)
            args.state_dir.mkdir(parents=True, exist_ok=True)
            with (args.state_dir / "collect.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(
                        "another CI collector holds the state lock"
                    ) from None
                store = Store(args.state_dir)
                config = store.configure(config)
                now = datetime.now(UTC).isoformat()
                result = collect(GitHub(args.repo, store, Path.cwd()), config, now)
            print(json.dumps(result, ensure_ascii=False))
            return 1 if result["errors"] else 0
        store = Store(args.state_dir, readonly=True)
        store.db.execute("BEGIN")
        if args.command == "records":
            if args.after < 0 or not 1 <= args.limit <= 1000:
                raise ValueError("--after must be nonnegative; --limit must be 1..1000")
            result = store.records(
                args.after, args.limit, sha(args.sha) if args.sha else None
            )
        elif args.command == "show":
            commit_sha = sha(args.sha)
            row = store.db.execute(
                "SELECT checked_at FROM commits WHERE sha=?", (commit_sha,)
            ).fetchone()
            if row is None:
                raise ValueError("SHA is outside the recorded main range")
            event = store.db.execute(
                "SELECT sequence, observed_at, payload FROM events WHERE sha=? ORDER BY sequence DESC LIMIT 1",
                (commit_sha,),
            ).fetchone()
            if event is None:
                raise ValueError("SHA observation is missing")
            result = {
                "schema_version": 1,
                "config": store.config(),
                "sequence": event[0],
                "observed_at": event[1],
                "checked_at": row[0],
                "observation": json.loads(event[2]),
            }
        else:
            if args.start_line < 1 or not 1 <= args.limit <= 1000:
                raise ValueError(
                    "--start-line must be positive; --limit must be 1..1000"
                )
            lines = store.object(args.object).splitlines()
            start = args.start_line - 1
            result = {
                "object": args.object,
                "start_line": args.start_line,
                "lines": lines[start : start + args.limit],
                "total_lines": len(lines),
                "has_more": start + args.limit < len(lines),
            }
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"ci_watch: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
