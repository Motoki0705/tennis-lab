"""Root-only Claude receiver lifecycle and unread context hooks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.identity import session_identity  # noqa: E402

from board import display, receivers  # noqa: E402
from board.errors import BoardError  # noqa: E402


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        event = payload.get("hook_event_name")
        if event not in {"SessionStart", "UserPromptSubmit"}:
            raise BoardError("unsupported board hook event")
        sid = payload.get("session_id")
        if not isinstance(sid, str) or not sid or payload.get("agent_id"):
            raise BoardError("board hook requires a root session")
        env = dict(os.environ)
        if env.get("CODEX_SESSION_ID") or env.get("CODEX_THREAD_ID"):
            raise BoardError("board hook cannot run in a Codex child")
        env["CLAUDE_CODE_SESSION_ID"] = sid
        session_identity(env)
        native = receivers.hook_process(env)
        if event == "SessionStart":
            source = payload.get("source", "startup")
            if source in {"startup", "resume", "clear"}:
                receivers.capture_claude(env, source, native)
            elif source != "compact":
                raise BoardError("unsupported board session source")
        context = display.context(env)
        if context:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": event,
                            "additionalContext": context,
                        }
                    },
                    ensure_ascii=False,
                )
            )
        return 0
    except Exception:
        # All environment values and exception messages are private. In
        # particular JSON/OS errors can contain credentials or socket locations.
        print(
            "board hook: root receiver or board state unavailable/inconsistent",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
