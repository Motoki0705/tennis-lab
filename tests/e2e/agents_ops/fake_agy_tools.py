"""Stream events and real hook subprocesses for the finite agy CLI fakes."""

from __future__ import annotations

import json
import re
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


def run_command(
    command: str,
    *,
    skip_hook: bool = False,
    display_command: str | None = None,
    force_execution: bool = False,
    hook_step_offset: int = 0,
) -> dict[str, str]:
    return tool_attempt(
        "run_command",
        {"CommandLine": command},
        skip_hook=skip_hook,
        displayed={"CommandLine": display_command}
        if display_command is not None
        else None,
        force_execution=force_execution,
        hook_step_offset=hook_step_offset,
    )


def read_file(
    path: str,
    *,
    skip_hook: bool = False,
    force_execution: bool = False,
    hook_step_offset: int = 0,
    display_path: str | None = None,
) -> dict[str, str]:
    return tool_attempt(
        "read_file",
        {"Path": path, "StartLine": 1, "EndLine": 1},
        skip_hook=skip_hook,
        force_execution=force_execution,
        hook_step_offset=hook_step_offset,
        displayed={"Path": display_path} if display_path is not None else None,
    )


def tool_attempt(
    tool: str,
    parameters: dict[str, Any],
    *,
    skip_hook: bool = False,
    displayed: dict[str, Any] | None = None,
    force_execution: bool = False,
    hook_step_offset: int = 0,
) -> dict[str, str]:
    global _step
    _step += 1
    info: dict[str, Any] = {
        "name": tool,
        "parameters": parameters if displayed is None else displayed,
    }
    step: dict[str, Any] = {"step_index": _step, "state": "ACTIVE", "tool_info": info}
    event("step_update", {"step_update": step})
    config = json.loads((Path.cwd() / ".agents/hooks.json").read_text())
    hook = config["readonly-command-guard"]["PreToolUse"][0]
    assert re.fullmatch(hook["matcher"], tool)
    if skip_hook:
        decision = {"decision": "allow", "reason": "injected missing hook"}
    else:
        proc = subprocess.run(
            shlex.split(hook["hooks"][0]["command"]),
            input=json.dumps(
                {
                    "stepIdx": _step + hook_step_offset,
                    "toolCall": {"name": tool, "args": parameters},
                }
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        decision = json.loads(proc.stdout)
    if decision["decision"] == "deny" and not force_execution:
        info["error"] = {
            "type": "synthetic_denial",
            "message": "Denied by pre-tool hook: " + decision["reason"],
        }
        step["state"] = "ERROR"
    else:
        # Commands are deliberately never executed by the fake.
        info["output"] = (
            "synthetic file contents"
            if tool == "read_file"
            else "Command exited with code 0"
        )
        step["state"] = "DONE"
    event("step_update", {"step_update": step})
    return decision


def emit_result(payload: dict[str, Any]) -> None:
    event("result", {"result": payload})
