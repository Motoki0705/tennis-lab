#!/usr/bin/env python3
"""Offline Claude stream for dispatcher tests; never calls an API or another agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value), flush=True)


def main() -> None:
    args = sys.argv[1:]
    task = Path(os.environ["AGENT_RUNTIME_RUN_DIR"]).parent
    input_dir = task / "agent-input"
    assert args[args.index("--add-dir") + 1] == str(input_dir)
    index = json.loads((input_dir / "index.json").read_text())
    response = Path(os.environ["FAKE_JUDGEMENT_RESPONSE"]).read_text()
    response = response.replace("INPUT_DIGEST", index["input_sha256"])
    response = response.replace("CODE_COMMIT", index["code_commit"])
    prompt = sys.stdin.read()
    with Path(os.environ["FAKE_JUDGEMENT_CALLS"]).open("a") as stream:
        stream.write(json.dumps({"args": args, "prompt": prompt}) + "\n")
    scenario = os.environ.get("FAKE_JUDGEMENT_SCENARIO", "success")
    model = args[args.index("--model") + 1]
    session = "private-judgement-session"
    emit(
        {
            "type": "system",
            "subtype": "init",
            "session_id": session,
            "model": "wrong-model" if scenario == "wrong_model" else model,
        }
    )
    if scenario == "cli_error":
        print("injected CLI failure", file=sys.stderr)
        raise SystemExit(1)
    if scenario == "empty":
        response = ""
    emit(
        {
            "type": "assistant",
            "session_id": session,
            "message": {"content": [{"type": "text", "text": response}]},
        }
    )
    emit(
        {
            "type": "result",
            "subtype": "success",
            "session_id": session,
            "is_error": False,
            "result": response,
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
    )


if __name__ == "__main__":
    main()
