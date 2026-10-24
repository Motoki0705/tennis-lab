#!/usr/bin/env python3
"""Offline CLI with a controllable completion event, for managed runtime tests."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value), flush=True)


def main() -> None:
    args = sys.argv[1:]
    sys.stdin.read()
    agent = Path(sys.argv[0]).name
    directory = Path(os.environ["AGENT_RUNTIME_RUN_DIR"])
    session = (
        "private-"
        + hashlib.sha256(os.environ["AGENT_RUNTIME_TASK_ID"].encode()).hexdigest()[:24]
    )
    if "--resume" in args:
        assert args[args.index("--resume") + 1] == session
    model = args[args.index("--model") + 1]
    with Path(os.environ["FAKE_DISPATCH_CALLS"]).open("a") as stream:
        stream.write(
            json.dumps({"args": args, "agent": agent, "directory": str(directory)})
            + "\n"
        )
    if agent == "claude":
        emit(
            {"type": "system", "subtype": "init", "session_id": session, "model": model}
        )
    else:
        emit({"type": "thread.started", "thread_id": session})
        transcript = Path(os.environ["CODEX_HOME"]) / "sessions" / (session + ".jsonl")
        transcript.parent.mkdir(parents=True, exist_ok=True)
        with transcript.open("a") as stream:
            stream.write(
                json.dumps({"type": "session_meta", "payload": {"id": session}}) + "\n"
            )
            stream.write(
                json.dumps(
                    {
                        "type": "turn_context",
                        "payload": {
                            "turn_id": str(uuid.uuid4()),
                            "cwd": str(Path.cwd()),
                            "model": model,
                            "effort": "max",
                        },
                    }
                )
                + "\n"
            )
        emit({"type": "turn.started"})
    if os.environ.get("FAKE_DISPATCH_HOLD"):
        while not (directory.parent / "release").exists():
            time.sleep(0.02)
    if agent == "claude":
        final = json.dumps(
            {
                "schema_version": 1,
                "state": "waiting",
                "reason": "CI待ち",
                "resume_condition": "CI完了通知",
            }
        )
        emit(
            {
                "type": "assistant",
                "session_id": session,
                "message": {"content": [{"type": "text", "text": final}]},
            }
        )
        emit(
            {
                "type": "result",
                "subtype": "success",
                "session_id": session,
                "is_error": False,
                "result": final,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
    else:
        final = "Implementation returned; parent must verify artifacts."
        Path(args[args.index("--output-last-message") + 1]).write_text(final)
        emit(
            {"type": "item.completed", "item": {"type": "agent_message", "text": final}}
        )
        emit(
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}
        )


if __name__ == "__main__":
    main()
