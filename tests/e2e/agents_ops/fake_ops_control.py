"""Stateful systemctl/journalctl/dispatcher substitutes for ops control e2e."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

state_file = Path(os.environ["FAKE_OPS_STATE"])
state = json.loads(state_file.read_text())
tool = Path(sys.argv[0]).name
if tool == "__main__.py":
    tool = "dispatcher"
args = sys.argv[1:]
call = [tool, *args]
with Path(os.environ["FAKE_OPS_CALLS"]).open("a") as stream:
    stream.write(json.dumps(call) + "\n")


def finish(code: int = 0) -> None:
    if code == 0 and call == state.get("lose_bus_after_call"):
        state["bus_lost"] = True
        del state["lose_bus_after_call"]
    state_file.write_text(json.dumps(state))
    raise SystemExit(code)


if tool == "journalctl":
    assert args[0] == "--user"
    print("fake journal: " + " ".join(args))
    finish(state.get("journal_exit", 0))

if tool == "dispatcher":
    command = args[0]
    if command in state.get("dispatcher_fail", []):
        print("fake dispatcher failure", file=sys.stderr)
        finish(7)
    if command == "enable":
        state["dispatcher"]["enabled"] = True
    elif command == "shutdown":
        state["dispatcher"] = state.get(
            "after_shutdown",
            {
                "schema_version": 1,
                "enabled": False,
                "parents": [],
                "active_judgement": None,
            },
        )
    elif command == "status":
        print(json.dumps(state["dispatcher"]))
        finish()
    else:
        raise AssertionError(args)
    print(json.dumps({"result": command}))
    finish()

assert tool == "systemctl" and args.pop(0) == "--user", (tool, args)
if state.get("manager") == "bus_unreachable" or state.get("bus_lost"):
    if state.get("bus_lost") and "bus_failures_remaining" in state:
        state["bus_failures_remaining"] -= 1
        state["bus_lost"] = state["bus_failures_remaining"] > 0
    print("Failed to connect to bus: No medium found", file=sys.stderr)
    finish(1)
if state.get("manager") == "manager_missing":
    # No manager property is NOT evidence of disabled units, even on rc=0.
    finish()
command = args[0]
if command == "show":
    if "--property=Version" in args:
        print("255.fake")
        finish()
    unit = args[1]
    data = state["units"][unit]
    for key, value in data.items():
        if key not in state.get("omit_properties", []):
            print(f"{key}={value}")
    finish(1 if data["LoadState"] == "not-found" else 0)
if command == "daemon-reload":
    for unit, data in state["units"].items():
        installed = Path(os.environ["XDG_CONFIG_HOME"]) / "systemd/user" / unit
        if installed.is_symlink():
            data["FragmentPath"] = str(installed)
    finish(state.get("reload_exit", 0))
if command == "link":
    for source in args[1:]:
        unit = Path(source).name
        assert Path(source).is_file(), source
        installed = Path(os.environ["XDG_CONFIG_HOME"]) / "systemd/user" / unit
        installed.parent.mkdir(parents=True, exist_ok=True)
        if not installed.is_symlink():
            installed.symlink_to(source)
        state["units"][unit]["FragmentPath"] = str(installed)
        state["units"][unit]["LoadState"] = "loaded"
        if unit not in state.get("sticky_enabled", []):
            state["units"][unit]["UnitFileState"] = "linked"
    finish()
assert command in {"enable", "disable", "start", "stop"}, args
for unit in args[1:]:
    if unit.startswith("--"):
        continue
    data = state["units"][unit]
    if f"{command}:{unit}" in state.get("fail_commands", []):
        print("fake unit operation failed", file=sys.stderr)
        finish(5)
    if command == "enable":
        assert "--now" in args
        data.update(UnitFileState="enabled", ActiveState="active")
    elif command == "disable":
        assert "--now" in args
        if unit not in state.get("sticky_enabled", []):
            data.update(UnitFileState="disabled")
            # systemctl disable removes the manual installation link, too.
            data["LoadState"] = "not-found"
            installed = Path(os.environ["XDG_CONFIG_HOME"]) / "systemd/user" / unit
            if installed.is_symlink():
                installed.unlink()
        if unit not in state.get("sticky_active", []):
            data["ActiveState"] = "inactive"
    elif command == "stop" and unit not in state.get("sticky_active", []):
        data["ActiveState"] = "inactive"
    elif command == "start":
        # A completed oneshot returns inactive and preserves enablement.
        data.update(ActiveState="inactive", Result="success", ExecMainStatus="0")
if state.get("lose_bus_after_mutation"):
    state["bus_lost"] = True
finish()
