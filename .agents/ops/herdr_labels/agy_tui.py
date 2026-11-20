"""Launch a manually operated agy TUI with a terminal-bound label lifecycle."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FrameType
from typing import Any

from lib.agy_session import interactive_options
from lib.opsproc import process_identity
from parent.state import Store

from .client import Herdr
from .coordination import locked, owner_token, pane_record, read_record, state_root
from .service import AGY_LAUNCH_ENV, TOKEN_KEYS, clear_inbox


def clear_after_exit(
    env: Mapping[str, str], terminal: str, process: dict[str, Any]
) -> None:
    """Fence cleanup by the original terminal and exited process generation."""
    client = Herdr(env)
    pane = client.pane()
    if pane["terminal_id"] != terminal:
        return
    scope = owner_token(client.socket_path())
    path = pane_record(state_root(env) / "herdr-labels", scope, terminal)
    with locked(path.with_suffix(".lock")):
        record = read_record(path)
        if (
            record is None
            or record.get("processes")
            != [{"pid": process["pid"], "start": process["start_ticks"]}]
            or record.get("state") != "active"
        ):
            return

        def check() -> None:
            if (
                process_identity(process["pid"]) == process
                or read_record(path) != record
            ):
                raise ValueError("cannot clear a live or replaced agy display")

        check()
        pane = client.pane()
        if pane["terminal_id"] != terminal:
            return
        if pane.get("tokens", {}).get("tl_engine") == "agy":
            pane = clear_inbox(client, pane, check)
            client.publish(
                pane,
                agent=pane.get("agent"),
                display=None,
                tokens=dict.fromkeys(TOKEN_KEYS, ""),
                validate=check,
            )
        Store.write_json(path, {**record, "state": "ended"})


def launch(args: Sequence[str], env: dict[str, str]) -> int:
    options = tuple(interactive_options(args))
    if env.get("HERDR_ENV") != "1":
        os.execvpe("agy", ["agy", *args], env)
    terminal = Herdr(env).pane()["terminal_id"]
    existing = next(
        (value for name, value in reversed(options) if name == "settings"), None
    )
    settings = (
        json.loads(
            existing
            if existing.lstrip().startswith("{")
            else Path(existing).read_text()
        )
        if existing
        else {}
    )
    if not isinstance(settings, dict):
        raise ValueError("agy launch settings must be a JSON object")
    settings["statusLine"] = {
        "type": "command",
        "command": shlex.join(
            [
                sys.executable,
                "-I",
                str(Path(__file__).with_name("agy_hook.py")),
                "status-line",
            ]
        ),
        "stack_with_default": True,
    }
    env[AGY_LAUNCH_ENV] = "1"
    agent: subprocess.Popen[bytes] | None = None

    def forward(number: int, _frame: FrameType | None) -> None:
        # The foreground process group already receives terminal SIGINT.
        if agent is not None and number != signal.SIGINT:
            agent.send_signal(number)

    with TemporaryDirectory(prefix="agy-parent-") as temp:
        config = Path(temp) / "settings.json"
        config.write_text(json.dumps(settings), encoding="utf-8")
        command = [
            "agy",
            *(
                f"--{name}" if value is None else f"--{name}={value}"
                for name, value in options
                if name != "settings"
            ),
            "--settings",
            str(config),
        ]
        handlers = {
            number: signal.signal(number, forward)
            for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            agent = subprocess.Popen(command, env=env)
            process = process_identity(agent.pid)
            result = agent.wait()
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
        if process is not None:
            clear_after_exit(env, terminal, process)
    return result if result >= 0 else 128 - result


def main() -> int:
    try:
        return launch(sys.argv[1:], dict(os.environ))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError):
        print(
            "agy parent launcher: could not start the TUI or verify display cleanup",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
