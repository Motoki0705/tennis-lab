#!/usr/bin/env python3
"""Receive explicit clear notifications from the patched Codex TUI, never its daemon."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parent.state import Store  # noqa: E402

from herdr_labels.coordination import state_root  # noqa: E402
from herdr_labels.service import sync  # noqa: E402


def main() -> int:
    env = dict(os.environ)
    try:
        payload = json.load(sys.stdin)
        if (
            not isinstance(payload, dict)
            or type(payload.get("version")) is not int
            or payload["version"] != 1
            or payload.get("event") != "clear"
        ):
            raise ValueError("expected a version 1 terminal clear event")
        sid = payload.get("session_id")
        if not isinstance(sid, str) or str(UUID(sid)) != sid:
            raise ValueError("terminal clear requires the new root thread UUID")
        if env.get("HERDR_ENV") != "1":
            print(json.dumps({"status": "not_configured", "reason": "outside Herdr"}))
            return 0
        # The new root thread's actual ID comes directly from thread/start's response.
        # Inherited IDs may still describe the preceding conversation.
        env["CODEX_SESSION_ID"] = env["CODEX_THREAD_ID"] = sid
        result = sync(env, clear=True, session_start=True, bind_native=True)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"herdr-label terminal clear: {message}", file=sys.stderr)
        try:
            Store.write_json(
                state_root(env) / "herdr-labels/terminal-clear-error.json",
                {
                    "time": datetime.now(UTC).isoformat(),
                    "error": message,
                },
            )
        except (ValueError, OSError) as log_error:
            print(
                f"could not record terminal clear error: {log_error}", file=sys.stderr
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
