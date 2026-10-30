#!/usr/bin/env python3
"""Offline Codex stream plus transcript for the real proposal-organization runner."""

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
    run_dir = Path(os.environ["AGENT_RUNTIME_RUN_DIR"])
    fixed = json.loads((run_dir.parent / "organization-input.json").read_text())
    plan_path = Path(os.environ["FAKE_ORGANIZATION_PLAN"])
    response = (
        plan_path.read_text()
        if plan_path.exists()
        else json.dumps(
            {
                "schema_version": 1,
                "input_sha256": "INPUT_DIGEST",
                "nodes": [],
                "decisions": [
                    {
                        "proposal": name,
                        "action": "hold",
                        "target": None,
                        "reason": "評価対象の選択が不足している。",
                        "evidence": [f"issue:{row['source']['report_number']}"],
                        "resume_condition": "評価対象をレポートへ追記する。",
                        "next_action": "不足情報を整理して再実行する。",
                    }
                    for name, row in fixed["proposals"].items()
                ],
            }
        )
    )
    response = response.replace("INPUT_DIGEST", fixed["input_sha256"])
    response = response.replace("CODE_COMMIT", fixed["snapshot"]["code"]["commit"])
    prompt = sys.stdin.read()
    with Path(os.environ["FAKE_ORGANIZATION_CALLS"]).open("a") as stream:
        stream.write(json.dumps({"args": args, "prompt": prompt}) + "\n")
    model = args[args.index("--model") + 1]
    scenario = os.environ.get("FAKE_ORGANIZATION_SCENARIO", "success")
    if scenario == "cli_error":
        raise SystemExit(3)
    if scenario == "empty":
        response = ""
    session = "offline-organization-" + uuid.uuid4().hex
    directory = Path(os.environ["CODEX_HOME"]) / "sessions/2026/10/22"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"rollout-{session}.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": session}})
        + "\n"
        + json.dumps(
            {
                "type": "turn_context",
                "payload": {
                    "turn_id": uuid.uuid4().hex,
                    "cwd": str(Path.cwd()),
                    "model": "wrong-model" if scenario == "wrong_model" else model,
                    "effort": "max",
                },
            }
        )
        + "\n"
    )
    emit({"type": "thread.started", "thread_id": session})
    emit({"type": "turn.started"})
    Path(args[args.index("--output-last-message") + 1]).write_text(response)
    emit(
        {
            "type": "item.completed",
            "item": {"id": "answer", "type": "agent_message", "text": response},
        }
    )
    emit({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 10}})


if __name__ == "__main__":
    main()
