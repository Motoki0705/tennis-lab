#!/usr/bin/env python3
"""Finite fake agy scenarios. Never delegate to a real agy installation."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["FAKE_AGY_ROOT"])
scenario = os.environ.get("FAKE_AGY_CASE", "success")
args = sys.argv[1:]
with (root / "invocations.jsonl").open("a") as trace:
    trace.write(
        json.dumps(
            {
                "args": args,
                "cwd": os.getcwd(),
                "disable_auto_update": os.environ.get("AGY_CLI_DISABLE_AUTO_UPDATE"),
                "read_environment": {
                    key: os.environ.get(key)
                    for key in (
                        "GIT_OPTIONAL_LOCKS",
                        "GIT_PAGER",
                        "GH_PAGER",
                        "RIPGREP_CONFIG_PATH",
                    )
                },
            }
        )
        + "\n"
    )
if args == ["--version"]:
    updated = (root / "updated").exists()
    print(
        "Antigravity CLI " + ("9.0.0" if scenario == "version" or updated else "1.4.11")
    )
    raise SystemExit(0)

if scenario == "timeout":
    child = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    (root / "child.pid").write_text(str(child.pid))
    signal.pause()
if scenario == "cli_error":
    print(
        'AGY_ERROR: {"error":"authentication failed; private-detail"}', file=sys.stderr
    )
    raise SystemExit(3)

prompt = args[args.index("-p") + 1]
probe = (
    json.loads(prompt.splitlines()[1])
    if prompt.startswith("AGY_PERMISSION_PROBE\n")
    else None
)
if probe and probe["stage"] == "deny" and scenario == "probe_cli_error":
    print('AGY_ERROR: {"error":"private-detail"}', file=sys.stderr)
    raise SystemExit(3)

log = Path(args[args.index("--log-file") + 1])
label = "Gemini 3.8 Flash (High)" if scenario != "model" else "Gemini 3.8 Pro"
if probe and probe["stage"] == "deny" and scenario == "probe_bad_model":
    label = "Other Model"
with log.open("a") as handle:
    if scenario != "missing_model":
        handle.write(
            f'Propagating selected model override to backend: label="{label}"\n'
        )
    if scenario == "conflicting_model":
        handle.write(
            'Propagating selected model override to backend: label="Other Model"\n'
        )
if scenario == "updated":
    (root / "updated").touch()

payload = json.loads((root / "payload.json").read_text())
if probe:
    work = Path.cwd()
    if probe["stage"] == "read":
        payload["structured_output"] = {
            "line": (work / "a").read_text().splitlines()[0],
            "subject": (work / ".git/COMMIT_EDITMSG").read_text().splitlines()[0],
        }
        payload["denied_actions"] = []
        if scenario == "probe_read_denied":
            payload.pop("structured_output")
            payload["denied_actions"] = [
                {"action": "command", "display_name": "RunCommand"}
            ]
    else:
        if scenario == "probe_workspace_write":
            (work / "unexpected").touch()
        elif scenario == "probe_outside_write":
            (work.parent / "outside/unexpected").touch()
        elif scenario == "probe_workspace_edit":
            (work / "a").write_text("changed by fake agent")
        payload.pop("structured_output")
        payload["denied_actions"] = [
            {"action": "command", "display_name": "RunCommand"}
        ]
        if scenario == "probe_missing_denial":
            payload["structured_output"] = {"attempted": True}
            payload["denied_actions"] = []
    print(json.dumps(payload))
    raise SystemExit(0)
if scenario == "empty":
    raise SystemExit(0)
if scenario == "no_structured":
    payload.pop("structured_output")
    payload["response"] = ""
    payload["denied_actions"] = [{"action": "command", "display_name": "RunCommand"}]
elif scenario == "schema":
    payload["structured_output"] = {"answer": 37}
elif scenario == "status":
    payload["status"] = "ERROR"
elif scenario == "finished_reason":
    payload["finished_reason"] = "print_timeout"
elif scenario == "malformed":
    print("not json")
    raise SystemExit(0)
elif scenario == "duplicate":
    print('{"status":"SUCCESS", "status":"ERROR"}')
    raise SystemExit(0)
elif scenario == "invalid_usage":
    payload["usage"] = {"input_tokens": -1}
elif scenario == "missing_metadata":
    payload.pop("usage")
    payload.pop("duration_seconds")
elif scenario == "wrong_headings":
    payload["structured_output"] = {"headings": ["incorrect"]}
print(json.dumps(payload, ensure_ascii=False))
