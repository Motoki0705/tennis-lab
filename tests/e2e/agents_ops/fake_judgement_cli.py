#!/usr/bin/env python3
"""Offline Codex stream for dispatcher tests; never calls an API or another agent."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value), flush=True)


def main() -> None:
    args = sys.argv[1:]
    task = Path(os.environ["AGENT_RUNTIME_RUN_DIR"]).parent
    input_dir = task / "agent-input"
    assert args[args.index("--sandbox") + 1] == "read-only"
    index = json.loads((input_dir / "index.json").read_text())
    response = Path(os.environ["FAKE_JUDGEMENT_RESPONSE"]).read_text()
    response = response.replace("INPUT_DIGEST", index["input_sha256"])
    response = response.replace("CODE_COMMIT", index["code_commit"])
    prompt = sys.stdin.read()
    with Path(os.environ["FAKE_JUDGEMENT_CALLS"]).open("a") as stream:
        stream.write(json.dumps({"args": args, "prompt": prompt}) + "\n")
    scenario = os.environ.get("FAKE_JUDGEMENT_SCENARIO", "success")
    if scenario == "schema_cli_error":
        print(
            "Error: --output-schema rejected schema "
            '"https://json-schema.org/draft/2020-12/schema"',
            file=sys.stderr,
        )
        raise SystemExit(1)
    model = args[args.index("--model") + 1]
    session = "private-judgement-session"
    home = Path(os.environ["CODEX_HOME"]) / "sessions"
    home.mkdir(parents=True, exist_ok=True)
    with (home / f"rollout-{session}.jsonl").open("w") as log:
        for kind, payload in (
            ("session_meta", {"id": session}),
            (
                "turn_context",
                {
                    "turn_id": str(uuid.uuid4()),
                    "cwd": str(Path.cwd()),
                    "model": "wrong-model" if scenario == "wrong_model" else model,
                    "effort": "xhigh",
                },
            ),
        ):
            log.write(json.dumps({"type": kind, "payload": payload}) + "\n")
    emit({"type": "thread.started", "thread_id": session})
    emit({"type": "turn.started"})
    if scenario == "reconnect":
        emit(
            {
                "type": "error",
                "message": "Reconnecting... 2/5 (workspace routing discovery timed out)",
            }
        )
    if scenario == "cli_error":
        print("injected CLI failure", file=sys.stderr)
        raise SystemExit(1)
    if scenario == "empty":
        response = ""
    if scenario != "missing_structured_output":
        Path(args[args.index("--output-last-message") + 1]).write_text(response)
    emit(
        {
            "type": "item.completed",
            "item": {
                "id": "item-1",
                "type": "agent_message",
                "text": response,
            },
        }
    )
    emit({"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 10}})


if __name__ == "__main__":
    main()
