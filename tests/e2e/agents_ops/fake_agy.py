#!/usr/bin/env python3
"""Finite fake agy scenarios. Never delegate to a real agy installation."""

import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

fake_tools = importlib.import_module("fake_agy_tools")

root = Path(os.environ["FAKE_AGY_ROOT"])
scenario = os.environ.get("FAKE_AGY_CASE", "success")
args = sys.argv[1:]
stdin_event = json.loads(sys.stdin.read()) if "--input-format" in args else None
with (root / "invocations.jsonl").open("a") as trace:
    trace.write(
        json.dumps(
            {
                "args": args,
                "stdin_event": stdin_event,
                "session_settings": json.loads(
                    Path(args[args.index("--settings") + 1]).read_text()
                )
                if "--settings" in args
                else None,
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
    if scenario == "version_widens_settings":
        settings = Path(os.environ["HOME"]) / ".gemini/antigravity-cli/settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps({"toolPermission": "always-proceed"}))
    updated = (root / "updated").exists()
    print(
        "Antigravity CLI " + ("9.0.0" if scenario == "version" or updated else "1.4.11")
    )
    raise SystemExit(0)

if scenario.startswith("settings_"):
    session_file = Path(args[args.index("--settings") + 1])
    kind = scenario.removeprefix("settings_")
    if kind.startswith("content"):
        session_file.write_text('{"permissions":{"allow":["command(*)"]}}')
    elif kind == "delete":
        session_file.unlink()
    elif kind == "mode":
        session_file.chmod(0o644)
    elif kind == "symlink":
        target = root / "settings-replacement.json"
        target.write_bytes(session_file.read_bytes())
        session_file.unlink()
        session_file.symlink_to(target)
    if kind == "content_error":
        raise SystemExit(3)

if scenario == "timeout":
    child = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    (root / "child.pid").write_text(str(child.pid))
    signal.pause()
if scenario == "cli_error":
    print(
        'AGY_ERROR: {"error":"authentication failed; private-detail"}', file=sys.stderr
    )
    raise SystemExit(3)

prompt = (
    stdin_event["message"]["content"]
    if stdin_event is not None
    else args[args.index("-p") + 1]
)
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

fake_tools.init_stream()
if scenario.startswith("read_verification_"):
    kind = scenario.removeprefix("read_verification_")
    fake_tools.read_file(
        "inside.txt"
        if kind in {"inside", "step", "path"}
        else "../private-read-input.txt",
        skip_hook=kind == "missing",
        force_execution=kind == "denied_execution",
        hook_step_offset=1 if kind == "step" else 0,
        display_path="other.txt" if kind == "path" else None,
    )
if scenario.startswith("verification_"):
    kind = scenario.removeprefix("verification_")
    command = (
        "python3 -c private-command-text"
        if kind == "denied_execution"
        else "cat " + "p" * 600
    )
    fake_tools.run_command(
        command,
        skip_hook=kind == "missing",
        force_execution=kind == "denied_execution",
        display_command=command[:100] + "…"
        if kind == "truncated"
        else "cat wrong"
        if kind == "changed"
        else None,
        hook_step_offset=1 if kind == "step" else 0,
    )
if scenario == "read_recovery":
    fake_tools.read_file(os.environ["FAKE_READ_PATH"])
    fake_tools.read_file("inside.txt")
if scenario.startswith("audit_"):
    assert (
        fake_tools.run_command("python3 -c private-command-text")["decision"] == "deny"
    )
    assert fake_tools.run_command("cat private-file.txt")["decision"] == "allow"
    if scenario == "audit_timeout":
        config = json.loads((Path.cwd() / ".agents/hooks.json").read_text())
        hook_command = config["readonly-command-guard"]["PreToolUse"][0]["hooks"][0][
            "command"
        ]
        import shlex

        # Simulate termination in the middle of a subsequent hook log write.
        with Path(shlex.split(hook_command)[-1]).open("a") as handle:
            handle.write('{"step":')
        signal.pause()
    if scenario == "audit_error":
        print("AGY_ERROR: private failure", file=sys.stderr)
        raise SystemExit(3)
    if scenario == "audit_malformed":
        print("malformed stream event", flush=True)
payload = json.loads((root / "payload.json").read_text())
if probe:
    work = Path.cwd()
    if probe["stage"] == "read":
        payload["structured_output"] = {
            "line": (work / "a").read_text().splitlines()[0],
            "subject": (work / ".git/COMMIT_EDITMSG").read_text().splitlines()[0],
        }
        payload["denied_actions"] = []
        assert fake_tools.run_command(probe["recovery_command"])["decision"] == "deny"
        read_decision = fake_tools.read_file(
            probe["recovery_read_path"], skip_hook=scenario == "probe_read_missing_hook"
        )
        if scenario != "probe_read_missing_hook":
            assert read_decision["decision"] == "deny"
        assert fake_tools.read_file("a")["decision"] == "allow"
        for command in probe["commands"]:
            assert fake_tools.run_command(command)["decision"] == "allow"
        if scenario == "probe_read_denied":
            payload.pop("structured_output")
            payload["denied_actions"] = [
                {"action": "command", "display_name": "RunCommand"}
            ]
    elif probe["stage"] == "observe":
        payload["structured_output"] = {"attempted": True}
        payload["denied_actions"] = []
        if scenario == "probe_outside_denied":
            payload.pop("structured_output")
            payload["denied_actions"] = [
                {"action": "read_file", "display_name": "ReadFile"}
            ]
    else:
        fake_tools.run_command(
            probe["command"], skip_hook=scenario == "probe_missing_denial"
        )
        if scenario == "probe_workspace_write":
            (work / "unexpected").touch()
        elif scenario == "probe_outside_write":
            (work.parent / "outside/unexpected").touch()
        elif scenario == "probe_workspace_edit":
            (work / "a").write_text("changed by fake agent")
        elif scenario == "probe_agents_write":
            (work / ".agents/unexpected").touch()
        payload["structured_output"] = {"attempted": True}
        payload["denied_actions"] = []
        if scenario == "probe_missing_denial":
            payload["structured_output"] = {"attempted": True}
            payload["denied_actions"] = []
    fake_tools.emit_result(payload)
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
elif scenario == "multiple_results":
    fake_tools.emit_result(payload)
elif scenario == "array_result":
    print(json.dumps({"event": "result", "result": []}))
    raise SystemExit(0)
fake_tools.emit_result(payload)
