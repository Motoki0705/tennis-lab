"""Antigravity's TUI status line and project PreInvocation entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from board.display import agy_unread  # noqa: E402
from board.errors import BoardError  # noqa: E402
from lib.agy_session import CONVERSATION_ENV, AgyTui, caller_tui  # noqa: E402

from herdr_labels.service import AGY_LAUNCH_ENV, agy_session_changed  # noqa: E402


def _event(
    env: dict[str, str], payload: dict[str, Any], mode: str
) -> tuple[AgyTui, str] | None:
    if mode == "pre-invocation":
        if (
            payload.get("hookEventName") != "PreInvocation"
            or type(payload.get("invocationNum")) is not int
            or payload["invocationNum"] != 0
        ):
            return None
        conversation = payload.get("conversationId")
        if conversation != env.get(CONVERSATION_ENV):
            return None
    elif mode == "status-line":
        if env.get("HERDR_ENV") != "1" or env.get(AGY_LAUNCH_ENV) != "1":
            return None
        conversation = payload.get("conversation_id")
        if conversation != payload.get("session_id"):
            return None
    else:
        raise ValueError("unexpected agy hook")
    if not isinstance(conversation, str) or not conversation:
        return None
    try:
        tui = caller_tui()
    except ValueError:
        return None  # Print, ACP and nested invocations also load project hooks.
    if tui.conversation() != conversation:
        return None  # Stale event or an in-process child conversation.
    return tui, conversation


def main() -> int:
    mode = sys.argv[1]
    env = dict(os.environ)
    reply: dict[str, Any] = {}
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("expected an agy hook object")
        event = _event(env, payload, mode)
        if event is not None:
            tui, conversation = event
            if mode == "status-line":
                agy_session_changed(env, tui, conversation)
            else:
                count = agy_unread(env, tui, conversation)
                if count:
                    reply = {
                        "injectSteps": [
                            {
                                "systemMessage": {
                                    "systemMessage": f"[board] 未読:{count}。作業の区切りで board inbox --unread を確認。"
                                }
                            }
                        ]
                    }
    except (
        BoardError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        KeyError,
    ):
        print(
            "agy parent hook: lifecycle or board state could not be verified",
            file=sys.stderr,
        )
        return 1
    if mode == "pre-invocation":
        print(json.dumps(reply, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
