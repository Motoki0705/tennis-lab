#!/usr/bin/env python3
"""Offline Claude/Codex CLI fixture; copied as an executable by runner tests."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value), flush=True)


def main() -> None:
    args = sys.argv[1:]
    Path(os.environ["FAKE_ARGS"]).write_text("\n".join(args))
    Path(os.environ["FAKE_CWD"]).write_text(str(Path.cwd()))
    if os.environ.get("FAKE_ENV"):
        keys = {
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDECODE",
            "CODEX_SESSION_ID",
            "CODEX_THREAD_ID",
            "AGENT_RUNTIME_RUN_DIR",
            "AGENT_RUNTIME_TASK_ID",
        }
        Path(os.environ["FAKE_ENV"]).write_text(
            json.dumps({k: v for k, v in os.environ.items() if k in keys})
        )
    agent = Path(sys.argv[0]).name
    scenario = os.environ.get("FAKE_SCENARIO", "success")
    if agent == "claude" and "--json-schema" in args:
        schema = json.loads(args[args.index("--json-schema") + 1])
        if (
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
            or scenario == "schema_cli_error"
        ):
            print(
                "Error: --json-schema is not a valid JSON Schema: no schema with key or ref "
                '"https://json-schema.org/draft/2020-12/schema"',
                file=sys.stderr,
            )
            sys.exit(1)
    model = args[args.index("--model") + 1]
    prompt = sys.stdin.read()
    session = "private-" + agent + "-session"
    if scenario == "wrong_session":
        session = "other-private-session"
    if agent == "codex":
        emit({"type": "thread.started", "thread_id": session})
        if scenario != "missing_transcript":
            home = Path(os.environ["CODEX_HOME"])
            directory = home / "sessions/2026/10/22"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"rollout-{session}.jsonl"
            context = {
                "turn_id": str(uuid.uuid4()),
                "cwd": str(Path.cwd()),
                "model": "unselected-model" if scenario == "wrong_model" else model,
                "effort": "low"
                if scenario == "wrong_effort"
                else "max"
                if 'model_reasoning_effort="max"' in args
                else "xhigh",
            }
            with path.open("a") as stream:
                stream.write(
                    json.dumps({"type": "session_meta", "payload": {"id": session}})
                    + "\n"
                )
                stream.write(
                    json.dumps({"type": "turn_context", "payload": context}) + "\n"
                )
                stream.write(
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": prompt}],
                            },
                        }
                    )
                    + "\n"
                )
                if scenario == "corrupt_transcript":
                    stream.write('{"type":"turn_context"}\n')
        emit({"type": "turn.started"})
    else:
        emit(
            {
                "type": "system",
                "subtype": "init",
                "session_id": session,
                "model": "unselected-model" if scenario == "wrong_model" else model,
            }
        )
    if scenario in {
        "hang",
        "leftover",
        "closed_stdout",
        "detached",
        "detached_leftover",
    }:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=scenario.startswith("detached"),
        )
        Path(os.environ["FAKE_CHILD"]).write_text(str(child.pid))
        if scenario not in {"leftover", "detached_leftover"}:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if scenario == "closed_stdout":
                sys.stdout.close()
                os.close(1)
            time.sleep(60)
    if scenario == "malformed":
        print("not JSON", flush=True)
        return
    if scenario == "duplicate_keys":
        print('{"type":"turn.completed","type":"turn.failed"}', flush=True)
        return
    if scenario == "quota":
        emit({"type": "turn.failed", "error": {"message": "usage limit exceeded"}})
        return
    if scenario == "exit_error":
        print("configured model unavailable", file=sys.stderr)
        sys.exit(7)
    structured = "--json-schema" in args or "--output-schema" in args
    payload = {"message": prompt.strip()}
    if scenario == "schema_mismatch":
        payload = {"message": 123}
    final = json.dumps(payload) if structured else prompt
    if scenario == "schema_preamble":
        final = "I'm writing the final JSON now.\n\n" + final
    if agent == "codex":
        out = Path(args[args.index("--output-last-message") + 1])
        if scenario != "schema_missing":
            out.write_text(
                "different final" if scenario == "different_final" else final
            )
        emit(
            {
                "type": "item.completed",
                "item": {"id": "item-0", "type": "agent_message", "text": final},
            }
        )
        if scenario != "missing_completion":
            emit(
                {
                    "type": "turn.completed",
                    "usage": None
                    if scenario == "no_usage"
                    else {
                        "input_tokens": -1 if scenario == "invalid_usage" else 10,
                        "output_tokens": 5,
                    },
                }
            )
    else:
        output = {}
        if structured and scenario != "schema_missing":
            output["structured_output"] = None if scenario == "schema_null" else payload
        emit(
            {
                "type": "result",
                "subtype": "error_max_structured_output_retries"
                if scenario == "schema_retries"
                else "success",
                "session_id": session,
                "is_error": scenario == "schema_retries",
                "result": "I'm writing the final JSON now.\n\n" + final
                if structured
                else final,
                **output,
                "usage": None
                if scenario == "no_usage"
                else {
                    "input_tokens": -1 if scenario == "invalid_usage" else 10,
                    "output_tokens": 5,
                },
                "modelUsage": {model: {"inputTokens": 10}},
                "total_cost_usd": 0.01,
            }
        )


if __name__ == "__main__":
    main()
