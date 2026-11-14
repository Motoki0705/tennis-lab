"""Local storage tier CLI. Invoke through checkout_job.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    from lib.checkout_lock import require_checkout_job

    require_checkout_job()

from lib.checkout_lock import CHECKOUT_ROOT_ENV, common_directory

from .engine import Engine
from .filesystem import StorageError, real_directory, write_json
from .layout import Config, main_checkout, prepare_worktree
from .usage import Inspector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument(
        "--queue-dir", type=Path, help="explicit nondefault shared training queue"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", help="read-only boundary and placement inventory")
    for name in ("mirror", "verify", "index", "hydrate", "evict", "cycle"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "units",
            nargs="+" if name in {"index", "hydrate"} else "*",
            help="repo-relative unit or a path inside it",
        )
        sub.add_argument("--dry-run", action="store_true")
    commands.add_parser(
        "recover", help="restore interrupted D: quarantines, never resume deletion"
    )
    activate = commands.add_parser(
        "activate", help="enable missing-input diagnostics after migration"
    )
    activate.add_argument("--dry-run", action="store_true")
    worktree = commands.add_parser(
        "prepare-worktree", help="create D: outputs and shared input/venv links"
    )
    worktree.add_argument("worktree", type=Path)
    worktree.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        repo = main_checkout(args.repo)
        if common_directory(repo) != common_directory(
            Path(os.environ[CHECKOUT_ROOT_ENV])
        ):
            raise StorageError("--repo must share the launcher's Git directory")
        config = Config.load(repo, args.config)
        dry_run = getattr(args, "dry_run", False)
        engine = Engine(
            config, inspector=Inspector(args.queue_dir or repo / ".training_queue")
        )
        if args.command == "prepare-worktree":
            report = prepare_worktree(config, args.worktree, dry_run=dry_run)
            code = 0
        elif args.command == "activate":
            with engine.locked(dry_run=dry_run):
                for name in ("data", "outputs", "ckpt"):
                    real_directory(repo / name, repo)
                marker = {
                    "schema_version": 1,
                    "repository": str(repo),
                    "mirror_mount": str(config.mount),
                }
                if not dry_run:
                    write_json(config.state / "active.json", marker)
                report = {
                    "action": "would_activate" if dry_run else "activated",
                    "dry_run": dry_run,
                }
                code = 0
        else:
            code, report = engine.run(
                args.command, getattr(args, "units", []), dry_run=dry_run
            )
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return code
    except (OSError, ValueError, KeyError, TypeError, StorageError) as exc:
        print(
            json.dumps({"outcome": "failure", "error": str(exc)}, ensure_ascii=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
