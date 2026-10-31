#!/usr/bin/env python3
"""Finite writer/verifier turns; never executes another agent."""

import json
import os
import sys
from pathlib import Path

root = Path(os.environ["REWRITE_FAKE_ROOT"])
args = sys.argv[1:]
if args == ["--version"]:
    print("Antigravity CLI 1.4.11")
    raise SystemExit(0)
with (root / "agy-calls.jsonl").open("a") as trace:
    trace.write(json.dumps(args) + "\n")
index = len((root / "agy-calls.jsonl").read_text().splitlines()) - 1
outputs = json.loads((root / "turns.json").read_text())
assert index < len(outputs), "unexpected additional agent call"
log = Path(args[args.index("--log-file") + 1])
log.write_text(
    'Propagating selected model override to backend: label="Gemini 3.8 Flash (High)"\n'
    "private-log conversation_id=private-conversation /home/private-person/token.json\n"
)
print(
    json.dumps(
        {
            "status": "SUCCESS",
            "conversation_id": "private-conversation",
            "duration_seconds": 0.2,
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            **outputs[index],
        },
        ensure_ascii=False,
    )
)
