"""spin worktrees scan / cleanup [--dry-run]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cleanup import Cleaner
from .storage import Busy, CleanupError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "cleanup"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-dir", type=Path)
        if name == "cleanup":
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        code, report = Cleaner(args.repo, directory=args.state_dir).run(
            cleanup=args.command == "cleanup", dry_run=getattr(args, "dry_run", False)
        )
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return code
    except (OSError, ValueError, CleanupError) as exc:
        print(json.dumps({"outcome": "failure", "error": str(exc)}), file=sys.stderr)
        return 3 if isinstance(exc, Busy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
