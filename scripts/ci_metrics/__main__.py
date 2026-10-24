"""Entrypoints for test observations and offline Actions attempt accounting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .actions import fetch_attempts, markdown, summarize_attempts
from .report import summarize
from .runner import environment_metadata, run, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="Run current CPU target, recording all workers")
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--workers", type=int, default=2)
    runner.add_argument(
        "--lock", type=Path, help="Absolute host-wide pytest lock (local only)"
    )
    runner.add_argument("pytest_args", nargs=argparse.REMAINDER)
    report = sub.add_parser(
        "report", help="Rebuild a summary, including interrupted runs"
    )
    report.add_argument("--output", required=True, type=Path)
    capture = sub.add_parser(
        "capture", help="Record runner before dependency preparation"
    )
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--workers", type=int, default=2)
    actions = sub.add_parser(
        "actions", help="Summarize saved attempt-specific REST responses"
    )
    actions.add_argument("--input", required=True, type=Path)
    actions.add_argument("--output", required=True, type=Path)
    actions.add_argument("--check", action="append", default=None)
    fetch = sub.add_parser(
        "fetch-actions", help="Save attempt-specific GitHub REST data (GET only)"
    )
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--run-id", required=True, type=int)
    fetch.add_argument("--attempt", type=int)
    fetch.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if args.workers < 0:
            parser.error("--workers must be nonnegative")
        state_root = Path(
            os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))
        )
        lock: Path | None = args.lock or Path(
            os.environ.get(
                "PYTEST_LOCK", str(state_root / "tennis-lab-agents/pytest.lock")
            )
        )
        # Actions runners are isolated hosts. Local callers always share the lock.
        if os.environ.get("GITHUB_ACTIONS") == "true" and args.lock is None:
            lock = None
        extra = args.pytest_args
        if extra[:1] == ["--"]:
            extra = extra[1:]
        return run(args.output, args.workers, extra, lock)
    if args.command == "report":
        args.output.mkdir(parents=True, exist_ok=True)
        result = summarize(args.output)
        print((args.output / "summary.md").read_text())
        return 0 if result["successful_test_baseline"] else 2
    if args.command == "capture":
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "environment.json", environment_metadata(args.workers, [])
        )
        return 0
    if args.command == "fetch-actions":
        records = fetch_attempts(args.repo, args.run_id, args.output, args.attempt)
        result = summarize_attempts(records, ["Python"])
        write_json(args.output / "actions-summary.json", result)
        (args.output / "actions-summary.md").write_text(markdown(result))
        # Snapshot retrieval can succeed while the job is running. Its summary
        # still explicitly marks incomplete timestamps and non-successful checks.
        return 0
    result = summarize_attempts(
        json.loads(args.input.read_text()), args.check or ["Python"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "actions-summary.json", result)
    document = markdown(result)
    (args.output / "actions-summary.md").write_text(document)
    print(document)
    return (
        2
        if result["missing_attempts"]
        or any(not row["complete_measurement"] for row in result["attempts"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
