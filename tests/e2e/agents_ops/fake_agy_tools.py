"""Stream events and real hook subprocesses for the finite agy CLI fakes."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

_step = 0


def event(kind: str, value: dict[str, Any]) -> None:
    print(json.dumps({"event": kind, **value}, ensure_ascii=False), flush=True)


def init_stream() -> None:
    event(
        "init",
        {
            "conversation_id": "private-session",
            "init": {"permission_mode": "request-review"},
        },
    )


def run_command(command: str, *, skip_hook: bool = False) -> dict[str, str]:
    global _step
    _step += 1
    info: dict[str, Any] = {
        "name": "run_command",
        "parameters": {"CommandLine": command},
    }
    step: dict[str, Any] = {"step_index": _step, "state": "ACTIVE", "tool_info": info}
    event("step_update", {"step_update": step})
    config = json.loads((Path.cwd() / ".agents/hooks.json").read_text())
    hook = config["readonly-command-guard"]["PreToolUse"][0]
    assert hook["matcher"] == "run_command"
    if skip_hook:
        decision = {"decision": "allow", "reason": "injected missing hook"}
    else:
        proc = subprocess.run(
            shlex.split(hook["hooks"][0]["command"]),
            input=json.dumps(
                {"stepIdx": _step, "toolCall": {"args": {"CommandLine": command}}}
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        decision = json.loads(proc.stdout)
    if decision["decision"] == "deny":
        info["error"] = {"message": "Denied by pre-tool hook: " + decision["reason"]}
        step["state"] = "ERROR"
    else:
        # Commands are deliberately never executed by the fake.
        info["output"] = "Command exited with code 0"
        step["state"] = "DONE"
    event("step_update", {"step_update": step})
    return decision


def emit_result(payload: dict[str, Any]) -> None:
    event("result", {"result": payload})
