"""Lock before importing an operational job from the mutable main checkout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.checkout_lock import checkout_lock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    resident = parser.add_mutually_exclusive_group()
    resident.add_argument("--supervisor", action="store_true")
    resident.add_argument("--board-web-port", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command and not args.supervisor and args.board_web_port is None:
        parser.error("a command after -- is required")
    if args.board_web_port is not None and not 1 <= args.board_web_port <= 65535:
        parser.error("board web port must be between 1 and 65535")
    with checkout_lock(args.repo) as fd:
        if args.supervisor:
            from dispatcher.cli import main as supervise
        elif args.board_web_port is not None:
            from board.web import serve
        else:
            assert fd is not None
            # exec preserves the lock across the actual job and inherited children.
            os.set_inheritable(fd, True)
            os.execvpe(command[0], command, os.environ)
    if args.board_web_port is not None:
        serve(os.environ, port=args.board_web_port)
        return 0
    return int(supervise(["supervise", *command]))


if __name__ == "__main__":
    raise SystemExit(main())
