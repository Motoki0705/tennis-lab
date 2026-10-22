"""Claude's real SessionStart/SessionEnd payload supplies its own session identity."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parent.state import session_identity  # noqa: E402

from herdr_labels.service import sync  # noqa: E402


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        event = payload.get("hook_event_name")
        if event not in {"SessionStart", "SessionEnd"}:
            raise ValueError("expected SessionStart or SessionEnd")
        sid = payload.get("session_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("Claude hook session_id is missing")
        env = dict(os.environ)
        if env.get("CODEX_SESSION_ID") or payload.get("agent_id"):
            raise ValueError("do not run parent display hooks from a child agent")
        env["CLAUDE_CODE_SESSION_ID"] = sid
        session_identity(env)
        if event == "SessionStart" and env.get("CLAUDE_ENV_FILE"):
            with Path(env["CLAUDE_ENV_FILE"]).open("a", encoding="utf-8") as stream:
                stream.write(f"\nexport CLAUDE_CODE_SESSION_ID={shlex.quote(sid)}\n")
        result = sync(
            env,
            clear=event == "SessionEnd",
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
        print(f"herdr-label hook: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
