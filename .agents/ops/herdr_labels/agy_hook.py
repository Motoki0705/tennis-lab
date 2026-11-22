"""Antigravity's manual-parent status line and PreInvocation entry points."""

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
from lib.agy_session import (  # noqa: E402
    AGY_LAUNCH_ENV,
    CONVERSATION_ENV,
    AgyExcluded,
    AgyTui,
    AgyVerificationError,
    caller_tui,
)

from herdr_labels.service import agy_session_changed  # noqa: E402


def _event(
    env: dict[str, str], payload: dict[str, Any], mode: str
) -> tuple[AgyTui, str] | None:
    if env.get(AGY_LAUNCH_ENV) != "1":
        return None
    if mode == "pre-invocation":
        if payload.get("hookEventName") != "PreInvocation":
            return None
        invocation = payload.get("invocationNum")
        if type(invocation) is not int or invocation < 0:
            raise AgyVerificationError("invalid agy hook invocation number")
        if invocation != 0:
            return None
        conversation = payload.get("conversationId")
        if not conversation or conversation != env.get(CONVERSATION_ENV):
            raise AgyVerificationError(
                "agy hook conversation differs from its environment"
            )
    elif mode == "status-line":
        if env.get("HERDR_ENV") != "1":
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
    except AgyExcluded:
        return None
    locked = tui.conversation()
    if locked is None:
        raise AgyVerificationError(
            "agy foreground TUI is missing its conversation lock"
        )
    if locked != conversation:
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
    except AgyVerificationError as exc:
        print(f"agy parent hook: {exc}", file=sys.stderr)
        return 1
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
