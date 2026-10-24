"""Parent board identity, messages and recipient state CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NoReturn

from . import messages
from .errors import BoardError
from .notifiers import configured, receipts
from .renaming import rename
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
        renaming = commands.add_parser(
            "rename", help="explicitly rename a registered parent"
        )
        renaming.add_argument("--parent", type=int, required=True)
        renaming.add_argument("--name")
        posting = commands.add_parser("post", help="append a post to board names")
        posting.add_argument("--to", action="append", required=True)
        posting.add_argument(
            "--kind", choices=("info", "question", "request"), required=True
        )
        posting.add_argument("--subject", required=True)
        posting.add_argument("--body-file", required=True)
        replying = commands.add_parser("reply", help="reply to the original author")
        replying.add_argument("--post", required=True)
        replying.add_argument("--to", action="append", default=[])
        replying.add_argument("--body-file", required=True)
        commands.add_parser(
            "read", help="retrieve a post and mark it read"
        ).add_argument("post_id")
        commands.add_parser(
            "thread", help="read a thread and mark addressed posts read"
        ).add_argument("post_id")
        commands.add_parser("inbox", help="list received post previews").add_argument(
            "--unread", action="store_true"
        )
        acknowledging = commands.add_parser("ack", help="acknowledge a received post")
        acknowledging.add_argument("--post", required=True)
        acknowledging.add_argument(
            "--state", choices=("accepted", "declined", "done"), required=True
        )
        commands.add_parser("status", help="show per-recipient states of sent posts")
        commands.add_parser(
            "receipts", help="refresh Codex queue receipt phases without sending"
        )
        viewing = commands.add_parser("web", help="serve the local read-only board")
        viewing.add_argument("--port", type=int, default=8790)
        args = parser.parse_args(argv)
        env = os.environ
        if args.command == "web":
            if not 1 <= args.port <= 65535:
                raise BoardError("board web port must be between 1 and 65535")
            # Web dependencies stay outside all other CLI commands.
            from .web import serve

            serve(env, port=args.port)
            return 0
        if args.command in {"post", "reply"}:
            with Path(args.body_file).open(encoding="utf-8") as source:
                body = source.read(60001)
            if args.command == "post":
                result = messages.post(
                    env,
                    to=args.to,
                    kind=args.kind,
                    subject=args.subject,
                    body=body,
                    notifiers=configured(env),
                )
            else:
                result = messages.reply(
                    env,
                    post_id=args.post,
                    to=args.to,
                    body=body,
                    notifiers=configured(env),
                )
        elif args.command == "rename":
            result = rename(env, parent=args.parent, name=args.name)
        elif args.command == "read":
            result = messages.read(env, args.post_id)
        elif args.command == "thread":
            result = messages.thread(env, args.post_id)
        elif args.command == "inbox":
            result = messages.inbox(env, unread=args.unread)
        elif args.command == "ack":
            result = messages.ack(env, post_id=args.post, state=args.state)
        elif args.command == "status":
            result = messages.status(env)
        elif args.command == "receipts":
            result = receipts(env)
        else:
            result = whoami(env) if args.command == "whoami" else agents(env)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if args.command == "rename" and result["display_sync"]["status"] == "error":
            return 1
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
