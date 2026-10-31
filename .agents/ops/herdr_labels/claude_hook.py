"""Claude's real SessionStart/SessionEnd payload supplies its own session identity."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.identity import session_identity  # noqa: E402

from herdr_labels.coordination import LifecycleCallerError  # noqa: E402
from herdr_labels.service import registry, sync  # noqa: E402


def main() -> int:
    event = None
    identity = None
    env = dict(os.environ)
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("expected a Claude hook payload object")
        event = payload.get("hook_event_name")
        if event not in {"SessionStart", "SessionEnd"}:
            raise ValueError("expected SessionStart or SessionEnd")
        sid = payload.get("session_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("Claude hook session_id is missing")
        if env.get("CODEX_SESSION_ID") or payload.get("agent_id"):
            raise ValueError("do not run parent display hooks from a child agent")
        env["CLAUDE_CODE_SESSION_ID"] = sid
        identity = session_identity(env)
        if event == "SessionStart" and env.get("CLAUDE_ENV_FILE"):
            with Path(env["CLAUDE_ENV_FILE"]).open("a", encoding="utf-8") as stream:
                stream.write(f"\nexport CLAUDE_CODE_SESSION_ID={shlex.quote(sid)}\n")
        if event == "SessionStart" and payload.get("source") == "compact":
            # Compaction continues this conversation; it is not an assignment
            # transition and must not restore or discard any existing label.
            print(
                json.dumps(
                    {
                        "herdr_label": {
                            "status": "preserved",
                            "reason": "compaction keeps the existing session display",
                        }
                    }
                )
            )
            return 0
        result = sync(
            env,
            clear=event == "SessionEnd"
            or (event == "SessionStart" and payload.get("source") == "clear"),
            session_start=event == "SessionStart",
            session_end=event == "SessionEnd",
        )
        print(json.dumps({"herdr_label": result}, ensure_ascii=False))
        return 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        notify_parent = False
        if (
            event == "SessionStart"
            and identity is not None
            and not isinstance(exc, LifecycleCallerError)
        ):
            try:
                notify_parent = any(
                    row["session"] == identity and row["status"] == "active"
                    for row in registry(env).values()
                )
            except (ValueError, OSError, KeyError, TypeError) as lookup_error:
                print(
                    f"herdr-label hook: cannot determine the active parent ({lookup_error}); "
                    f"original failure: {exc}",
                    file=sys.stderr,
                )
                return 1
        if notify_parent:
            reason = str(exc)
            message = (
                f"Herdrの担当表示の同期に失敗しました: {reason}。"
                ".agents/skills/herdr-session-label/SKILL.md の修復手順を確認してください。"
            )
            # SessionStart only delivers stdout JSON to Claude on exit zero.
            # This acknowledges delivery, while the payload reports sync failure.
            print(
                json.dumps(
                    {
                        "herdr_label": {"status": "error", "reason": reason},
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": message,
                        },
                        "systemMessage": message,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(f"herdr-label hook: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
