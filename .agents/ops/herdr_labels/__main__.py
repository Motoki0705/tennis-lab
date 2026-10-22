"""Self-service Herdr labels. Cross-session rollout is an explicit operator action."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .service import bind, sync


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser("sync")
    update.add_argument("--parent", type=int)
    update.add_argument("--topic")
    commands.add_parser("clear")
    setup = commands.add_parser(
        "bind", help="explicitly bind this session after inspecting the target"
    )
    setup.add_argument("--herdr-session", required=True)
    setup.add_argument("--pane", required=True)
    setup.add_argument("--terminal", required=True)
    args = parser.parse_args()
    try:
        env = dict(os.environ)
        if args.command == "bind":
            result = bind(env, args.herdr_session, args.pane, args.terminal)
        else:
            result = sync(
                env,
                clear=args.command == "clear",
                topic=getattr(args, "topic", None),
                expected_parent=getattr(args, "parent", None),
            )
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result["status"] == "not_configured" else 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"herdr-label: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
