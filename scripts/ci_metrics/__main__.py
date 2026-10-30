"""Entrypoints for test observations and offline Actions attempt accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .actions import fetch_attempts, markdown, summarize_attempts
from .profile import build_profile
from .report import summarize
from .runner import environment_metadata, run, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="Run current CPU target, recording all workers")
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--workers", type=int, default=2)
    runner.add_argument("--shard-index", type=int)
    runner.add_argument("--shard-count", type=int)
    runner.add_argument("--shard-weights", type=Path)
    runner.add_argument("--lane-plan", type=Path)
    runner.add_argument(
        "--lock", type=Path, help="Optional absolute lock for a coordinated comparison"
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
    profile = sub.add_parser(
        "profile", help="Build weights from complete shard artifacts"
    )
    profile.add_argument("--root", required=True, type=Path)
    profile.add_argument("--output", required=True, type=Path)
    profile.add_argument(
        "--split-file",
        action="append",
        help="Reviewed independent fixture file (repeatable)",
    )
    args = parser.parse_args()
    if args.command == "profile":
        result = build_profile(args.root, args.split_file)
        write_json(args.output, result)
        print(json.dumps(result["source"], indent=2))
        return 0
    if args.command == "run":
        if args.workers < 0:
            parser.error("--workers must be nonnegative")
        extra = args.pytest_args
        if extra[:1] == ["--"]:
            extra = extra[1:]
        shard_values = (args.shard_index, args.shard_count, args.shard_weights)
        if any(value is not None for value in shard_values) and any(
            value is None for value in shard_values
        ):
            parser.error(
                "--shard-index, --shard-count and --shard-weights are required together"
            )
        return run(
            args.output,
            args.workers,
            extra,
            args.lock,
            shard_values if args.shard_index is not None else None,
            args.lane_plan,
        )
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
