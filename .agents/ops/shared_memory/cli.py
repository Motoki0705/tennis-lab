"""Command line entry points for shared memory inspection and index syncing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

from .schema import iso_date
from .validate import Problem, audit, plan_sync, validate


def _today(value: str) -> date:
    try:
        return iso_date(value, "--today")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="validate shared memory")
    sync = commands.add_parser(
        "sync-index", help="regenerate SUMMARY.md and layer indexes"
    )
    inspect = commands.add_parser(
        "audit", help="report freshness and size usage as JSON"
    )
    for command in (check, sync, inspect):
        command.add_argument(
            "--repo",
            type=Path,
            default=Path.cwd(),
            help="repository root (default: current directory)",
        )
    sync.add_argument(
        "--check", action="store_true", help="report differences without writing files"
    )
    inspect.add_argument("--today", type=_today, help="audit date (YYYY-MM-DD)")
    return parser


def _print_problems(problems: Sequence[Problem]) -> None:
    for problem in problems:
        print(f"{problem.path}: [{problem.code}] {problem.message}", file=sys.stderr)


def _replace(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit":
            report = audit(args.repo, today=args.today)
            print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
            return int(bool(report.problems or report.stale))
        if args.command == "validate":
            problems = validate(args.repo)
            _print_problems(problems)
            return int(bool(problems))

        changes, problems = plan_sync(args.repo)
        if problems:
            _print_problems(problems)
            return 1
        for path, content in changes.items():
            if args.check:
                print(f"{path}: out of sync", file=sys.stderr)
            else:
                _replace(args.repo / path, content)
        return int(args.check and bool(changes))
    except OSError as exc:
        print(f"shared_memory: {exc}", file=sys.stderr)
        return 1
