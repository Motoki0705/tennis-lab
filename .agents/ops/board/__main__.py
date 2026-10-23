"""Read-only board identity and parent roster CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NoReturn

from .errors import BoardError
from .service import agents, whoami


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise BoardError("invalid board arguments; use board --help")


def main(argv: list[str] | None = None) -> int:
    try:
        parser = Parser(prog="board", description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("whoami", help="show the caller's active parent name")
        commands.add_parser("agents", help="list registered active parents")
        args = parser.parse_args(argv)
        result = whoami(os.environ) if args.command == "whoami" else agents(os.environ)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BoardError as exc:
        print(f"board: {exc}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError):
        # OS/JSON/registry errors can embed IDs, paths or arbitrary payloads.
        print(
            "board: configuration or parent registry unavailable/inconsistent",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
