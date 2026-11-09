#!/usr/bin/env python3
"""Executable agy protocol fixture; cannot delegate to a real model."""

import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

root = Path(os.environ["FAKE_EXPLAIN_ROOT"])
case = os.environ.get("FAKE_EXPLAIN_CASE", "success")
args = sys.argv[1:]
prompt = None
if "--input-format" in args:
    prompt = json.loads(sys.stdin.read())["message"]["content"]
with (root / "calls.jsonl").open("a") as trace:
    trace.write(json.dumps({"args": args, "prompt": prompt, "cwd": os.getcwd()}) + "\n")
if args == ["--version"]:
    print("Antigravity CLI 1.4.11")
    raise SystemExit(0)

log = Path(args[args.index("--log-file") + 1])
log.write_text(
    'Propagating selected model override to backend: label="'
    + ("Gemini 3.8 Flash (Medium)" if case == "model" else "Gemini 3.8 Flash (High)")
    + '"\n'
)
if "-p" in args:
    assert args[args.index("-p") + 1] == "/effort"
    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "num_turns": 0,
                "command": {
                    "name": "effort",
                    "data": {
                        "adjustable": True,
                        "current": "medium" if case == "effort" else "high",
                        "available": ["low", "medium", "high"],
                    },
                },
            }
        )
    )
    raise SystemExit(0)

schema = json.loads(Path(args[args.index("--json-schema") + 1]).read_text())
role = "writer" if schema["title"] == "Candidate" else "verifier"
if case == "hang" and role == "writer":
    hooks = json.loads((Path.cwd() / ".agents/hooks.json").read_text())
    command = hooks["explain-html-no-tools"]["PreToolUse"][0]["hooks"][0]["command"]
    denial = subprocess.run(
        shlex.split(command),
        input='{"toolCall":{"name":"run_command"}}',
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(denial.stdout)["decision"] == "deny"
    # Model a hook interrupted while journaling: it must not mask cancellation.
    with Path(shlex.split(command)[-3]).open("a") as journal:
        journal.write('{"input":')
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,signal,sys; os.write(int(sys.argv[1]),b'1'); signal.pause()",
            str(write_fd),
        ],
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    assert os.read(read_fd, 1) == b"1"
    os.close(read_fd)
    with (root / "ready").open("w") as ready:
        ready.write(json.dumps({"leader": os.getpid(), "tool": child.pid}))
    signal.pause()

settings = json.loads(Path(args[args.index("--settings") + 1]).read_text())
print(
    json.dumps(
        {
            "event": "init",
            "conversation_id": role,
            "init": {
                "model": "gemini-3.8-flash-high",
                "permission_mode": settings["toolPermission"],
            },
        }
    ),
    flush=True,
)
if case == "tool":
    hooks = json.loads((Path.cwd() / ".agents/hooks.json").read_text())
    entry = hooks["explain-html-no-tools"]["PreToolUse"][0]
    command = entry["hooks"][0]["command"]
    payload: dict[str, Any] = {
        "stepIdx": 1,
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "touch forbidden-public-file"},
        },
    }
    denial = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    if json.loads(denial.stdout).get("decision") != "deny":
        (root / "forbidden-public-file").touch()
    print(
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": role,
                    "step_index": 1,
                    "state": "ERROR",
                    "tool_info": {
                        "name": "run_command",
                        "parameters": payload["toolCall"]["args"],
                        "error": "Denied by pre-tool hook",
                    },
                },
            }
        ),
        flush=True,
    )
output = json.loads((root / (role + ".json")).read_text())
hooks = json.loads((Path.cwd() / ".agents/hooks.json").read_text())
command = hooks["explain-html-no-tools"]["PreToolUse"][0]["hooks"][0]["command"]
if role == "writer":
    # The real CLI can reject malformed finish arguments before invoking hooks.
    print(
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "conversation_id": role,
                    "step_index": 2,
                    "step_type": "tool",
                    "state": "ERROR",
                    "tool_info": {
                        "name": "finish",
                        "parameters": {},
                        "error": "invalid arguments",
                    },
                },
            }
        ),
        flush=True,
    )
completion = {"stepIdx": 4, "toolCall": {"name": "finish", "args": output}}
decision = subprocess.run(
    shlex.split(command),
    input=json.dumps(completion),
    text=True,
    capture_output=True,
    check=True,
)
allowed = json.loads(decision.stdout)["decision"] == "allow"
print(
    json.dumps(
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": role,
                "step_index": 4,
                "step_type": "finish",
                "state": "DONE" if allowed else "ERROR",
            },
        }
    ),
    flush=True,
)
print(
    json.dumps(
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "structured_output": output if allowed else None,
                "usage": {
                    "input_tokens": 101,
                    "output_tokens": 37,
                    "thinking_tokens": 11,
                },
            },
        }
    ),
    flush=True,
)
