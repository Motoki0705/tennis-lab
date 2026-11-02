"""Invocation-local PreToolUse settings backed by trusted controller code."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

# -I ignores cwd/PYTHONPATH, including modules planted in the model workspace.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_events import parse_json  # noqa: E402
from agy_permissions import command_allowed  # noqa: E402

DENIAL_REASON = (
    "This shell command is outside the permitted read-only forms. "
    "Continue using read_file or a permitted command and return the requested structured output."
)
READ_DENIAL_REASON = (
    "All task input is in the prompt. Files outside the workspace are unavailable. "
    "Continue with workspace files and return the requested structured output."
)


def install_command_hook(work: Path, record: Path) -> dict[Path, bytes]:
    """Only the settings enter the disposable workspace; never replace a hook."""
    work, record = work.resolve(), record.absolute()
    script = Path(__file__).resolve()
    if script.is_relative_to(work) or record.resolve().is_relative_to(work):
        raise ValueError("hook code and private records must be outside the workspace")
    agents = work / ".agents"
    if agents.is_symlink():
        raise ValueError("workspace .agents must not be a symlink")
    agents.mkdir(exist_ok=True, mode=0o700)
    command = shlex.join(
        [sys.executable, "-I", "-B", str(script), str(work), str(record)]
    )
    content = json.dumps(
        {
            "readonly-command-guard": {
                "PreToolUse": [
                    {
                        "matcher": "^(run_command|read_file)$",
                        "hooks": [{"type": "command", "command": command}],
                    }
                ]
            }
        }
    ).encode()
    hook_file = agents / "hooks.json"
    with hook_file.open("xb") as handle:
        hook_file.chmod(0o600)
        handle.write(content)
    return {hook_file: content}


def evaluate(raw: str, *, workspace: Path | None = None) -> dict[str, Any]:
    """Malformed tool input is a recoverable denial too, never an implicit allow."""
    entry: dict[str, Any] = {"step": None, "command": None, "decision": "deny"}
    try:
        payload = parse_json(raw)
        step = payload["stepIdx"]
        tool = payload["toolCall"]["name"]
        args = payload["toolCall"]["args"]
        if type(step) is not int or step < 0 or not isinstance(args, dict):
            raise ValueError("invalid hook input")
        entry.update(step=step, tool=tool)
        if tool == "run_command":
            command = args["CommandLine"]
            if not isinstance(command, str):
                raise ValueError("invalid command hook input")
            entry["command"] = command
            if command_allowed(command):
                entry["decision"] = "allow"
            entry["reason"] = (
                "Permitted read-only command."
                if entry["decision"] == "allow"
                else DENIAL_REASON
            )
        elif tool == "read_file":
            # Fixed CLI 1.4.11: toolCall.args.Path, StartLine, EndLine. The root
            # comes from the controller argv, never from untrusted hook input.
            path = args["Path"]
            if not isinstance(path, str) or not path.strip() or workspace is None:
                raise ValueError("invalid read_file hook input")
            entry["path"] = path
            root = workspace.resolve(strict=True)
            target = (root / Path(path).expanduser()).resolve()
            entry["resolved_path"] = str(target)
            if target.is_relative_to(root):
                entry["decision"] = "allow"
            entry["reason"] = (
                "Permitted workspace read."
                if entry["decision"] == "allow"
                else READ_DENIAL_REASON
            )
        else:
            raise ValueError("unsupported hook tool")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError):
        entry.update(reason="Malformed command hook input. " + DENIAL_REASON, input=raw)
    return entry


def main() -> None:
    entry = evaluate(sys.stdin.read(), workspace=Path(sys.argv[1]))
    try:
        # The controller creates this private file exclusively before starting agy.
        descriptor = os.open(sys.argv[2], os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except (OSError, IndexError):
        entry.update(
            decision="deny", reason="Command audit unavailable. " + DENIAL_REASON
        )
    print(json.dumps({key: entry[key] for key in ("decision", "reason")}))


if __name__ == "__main__":
    main()
