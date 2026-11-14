"""spin worktrees scan / cleanup [--dry-run]."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lib.checkout_lock import CHECKOUT_ROOT_ENV, common_directory

if __name__ == "__main__":
    from lib.checkout_lock import require_checkout_job

    require_checkout_job()

from .cleanup import Cleaner  # noqa: E402
from .storage import Busy, CleanupError  # noqa: E402


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
        cleaner = Cleaner(args.repo, directory=args.state_dir)
        if cleaner.common != common_directory(Path(os.environ[CHECKOUT_ROOT_ENV])):
            raise CleanupError(
                "--repo must share the launcher repository's Git directory"
            )
        code, report = cleaner.run(
            cleanup=args.command == "cleanup", dry_run=getattr(args, "dry_run", False)
        )
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return code
    except (OSError, ValueError, CleanupError) as exc:
        print(json.dumps({"outcome": "failure", "error": str(exc)}), file=sys.stderr)
        return 3 if isinstance(exc, Busy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
