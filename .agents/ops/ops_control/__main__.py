"""Install and control named systemd user jobs; never enable jobs during install."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .control import Controller
from .systemd import ControlError, ManagerUnavailable
from .targets import TARGETS, select


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spin ops", description=__doc__)
    parser.add_argument(
        "action", choices=("status", "install", "on", "off", "run", "logs")
    )
    parser.add_argument("target", choices=tuple(TARGETS), nargs="?")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preview install links without a user bus",
    )
    parser.add_argument(
        "--lines", type=int, default=100, help="journal lines (logs only)"
    )
    parser.add_argument(
        "--follow", action="store_true", help="follow journal (logs only)"
    )
    args = parser.parse_args(argv)
    if args.action not in {"status", "install"} and args.target is None:
        parser.error(f"{args.action} requires a target")
    if args.dry_run and args.action != "install":
        parser.error("--dry-run is only valid for install")
    if (args.follow or args.lines != 100) and args.action != "logs":
        parser.error("--follow/--lines are only valid for logs")
    if args.lines < 0:
        parser.error("--lines must be nonnegative")
    targets = select(args.target)
    if args.action == "run" and any(target.run_unit is None for target in targets):
        parser.error(
            "run requires a timer + oneshot target (dispatcher/agents cannot run once)"
        )
    controller = Controller(Path(__file__).resolve().parents[3])
    reports = []
    try:
        if args.action == "logs":
            return controller.logs(targets, lines=args.lines, follow=args.follow)
        if not args.dry_run:
            controller.systemd.require_manager()
        if args.action == "install":
            return controller.install(
                targets if args.target else (), dry_run=args.dry_run
            )
        for target in targets:
            reports.append(controller.operate(args.action, target))
    except ManagerUnavailable as exc:
        print(
            json.dumps(
                {
                    "manager": "unavailable",
                    "action": args.action,
                    "requested_target": args.target,
                    "outcome": "unavailable",
                    "error": str(exc),
                    "targets": reports,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    except (ControlError, OSError, ValueError) as exc:
        print(f"ops: {exc}", file=sys.stderr)
        return 1
    failed = any(report["errors"] for report in reports)
    print(
        json.dumps(
            {
                "manager": "available",
                "action": args.action,
                "outcome": "partial_failure" if failed else "ok",
                "targets": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
