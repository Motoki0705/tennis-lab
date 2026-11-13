"""Lock before importing an operational job from the mutable main checkout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.checkout_lock import (  # noqa: E402
    CHECKOUT_FD_ENV,
    CHECKOUT_ROOT_ENV,
    checkout_barrier,
    checkout_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    resident = parser.add_mutually_exclusive_group()
    resident.add_argument("--supervisor", action="store_true")
    resident.add_argument("--board-web-port", type=int)
    parser.add_argument("--state-dir", type=Path, help="supervisor state override")
    parser.add_argument(
        "--parent-state-dir", type=Path, help="supervisor registry override"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command and not args.supervisor and args.board_web_port is None:
        parser.error("a command after -- is required")
    if args.board_web_port is not None and not 1 <= args.board_web_port <= 65535:
        parser.error("board web port must be between 1 and 65535")
    if (args.state_dir or args.parent_state_dir) and not args.supervisor:
        parser.error("state overrides require --supervisor")
    with checkout_lock(args.repo) as fd:
        if checkout_barrier(args.repo).exists():
            print(
                "checkout_job: checkout recovery pending; inspect spin ops status git-sync",
                file=sys.stderr,
            )
            return 75
        if args.supervisor:
            from dispatcher.cli import main as supervise
        elif args.board_web_port is not None:
            from board.web import serve
        else:
            assert fd is not None
            # exec preserves the lock across the actual job and inherited children.
            os.set_inheritable(fd, True)
            environment = {
                **os.environ,
                CHECKOUT_ROOT_ENV: str(args.repo.resolve()),
                CHECKOUT_FD_ENV: str(fd),
            }
            os.execvpe(command[0], command, environment)
    if args.board_web_port is not None:
        serve(os.environ, port=args.board_web_port)
        return 0
    options: list[str] = []
    for name in ("state_dir", "parent_state_dir"):
        if getattr(args, name):
            options.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    return int(supervise([*options, "supervise", *command], _supervisor_launcher=True))


if __name__ == "__main__":
    raise SystemExit(main())
