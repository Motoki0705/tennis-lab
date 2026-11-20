"""Inspect, repair and publish shared memory."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

from .operations import load_json
from .publish import publish_operations
from .schema import SHARED_LAYERS, iso_date
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
    publication = []
    for name, description in (
        ("add", "publish a new entry"),
        ("update", "publish changes to an existing entry"),
        ("delete", "publish removal of an entry"),
        ("batch", "publish a batch as one commit"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument(
            "--spec", required=name != "delete", help="JSON file or - for stdin"
        )
        command.add_argument(
            "--remote", default="origin", help="configured GitHub remote"
        )
        command.add_argument(
            "--dry-run", action="store_true", help="validate and report the diff"
        )
        if name == "delete":
            command.add_argument("--layer", choices=SHARED_LAYERS)
            command.add_argument("--slug")
            command.add_argument("--reason")
        publication.append(command)
    for command in (check, sync, inspect, *publication):
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


def _spec(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[dict[str, object]]:
    if args.command == "delete":
        flags = {field: getattr(args, field) for field in ("layer", "slug", "reason")}
        if args.spec is None:
            if None in flags.values():
                parser.error(
                    "delete requires --spec or all of --layer, --slug and --reason"
                )
            return [{"op": "delete", **flags}]
        if any(value is not None for value in flags.values()):
            parser.error(
                "delete --spec cannot be combined with --layer, --slug or --reason"
            )
    raw = (
        sys.stdin.read()
        if args.spec == "-"
        else Path(args.spec).read_text(encoding="utf-8")
    )
    value = load_json(raw)
    if args.command == "batch":
        if not isinstance(value, list) or any(
            not isinstance(item, dict) for item in value
        ):
            raise ValueError("batch spec must be a JSON array of operation objects")
        return value
    if not isinstance(value, dict) or value.get("op", args.command) != args.command:
        raise ValueError("spec must be a JSON object for the selected command")
    return [{**value, "op": args.command}]


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
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"add", "update", "delete", "batch"}:
            result = publish_operations(
                args.repo, _spec(parser, args), remote=args.remote, dry_run=args.dry_run
            )
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return 0
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
    except (OSError, ValueError, RecursionError) as exc:
        print(f"shared_memory: {exc}", file=sys.stderr)
        return 1
