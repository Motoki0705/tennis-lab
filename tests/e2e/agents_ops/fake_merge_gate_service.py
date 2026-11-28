"""Test systemd-run boundary for the merge controller's transient unit."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


def run() -> int:
    options = sys.argv[1 : sys.argv.index("--")]
    command = sys.argv[sys.argv.index("--") + 1 :]
    assert {"--user", "--collect", "--service-type=notify"} <= set(options)
    # Pretend the manager still has old authorization and source settings.
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(
            (
                "AGENTS_TEST_",
                "PARENT_GH_",
                "TENNIS_TEST_",
                "MERGE_GATE_TEST_",
                "REVIEW_AGENT_",
            )
        )
    }
    env.update(AGENT_AUTO_MERGE="1", PYTHONPATH="/stale/controller/source")
    properties: dict[str, str] = {}
    pre: list[list[str]] = []
    post: list[list[str]] = []
    for flag in options:
        kind, _, value = flag.partition("=")
        if kind == "--property":
            name, _, setting = value.partition("=")
            if name in ("ExecStartPre", "ExecStopPost"):
                assert setting.startswith(":")
                (pre if name == "ExecStartPre" else post).append(
                    shlex.split(setting[1:])
                )
            else:
                properties[name] = setting
        elif kind == "--setenv":
            name, supplied, setting = value.partition("=")
            env[name] = setting if supplied else os.environ[name]
    for name in properties["UnsetEnvironment"].split():
        env.pop(name, None)
    cwd = Path(properties["WorkingDirectory"].removeprefix("-"))
    log_path = Path(properties["StandardOutput"].removeprefix("append:"))
    assert properties["StandardError"] == properties["StandardOutput"]
    checkout = Path(command[1]).parents[3]
    with Path(env["MERGE_GATE_TEST_LAUNCHES"]).open("a") as record:
        record.write(
            json.dumps(
                {
                    "checkout": str(checkout),
                    "unit": next(
                        flag.removeprefix("--unit=")
                        for flag in options
                        if flag.startswith("--unit=")
                    ),
                    "env": {
                        name: env.get(name)
                        for name in (
                            "AGENT_AUTO_MERGE",
                            "CODEX_HOME",
                            "GH_CONFIG_DIR",
                            "XDG_CONFIG_HOME",
                            "XDG_STATE_HOME",
                            "TRAINING_QUEUE_DIR",
                            "PARENT_LOCAL_JOBS_DIR",
                            "PYTHONPATH",
                            "PATH",
                        )
                    },
                }
            )
            + "\n"
        )
    # Only the transport and capacity test shims are added to the service.
    env["PYTHONPATH"] += os.pathsep + env["MERGE_GATE_TEST_BOOTSTRAP"]
    with (
        tempfile.TemporaryDirectory(prefix="gate-ready-") as short,
        socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as ready,
        log_path.open("ab") as log,
    ):
        env["NOTIFY_SOCKET"] = str(Path(short) / "ready")
        ready.bind(env["NOTIFY_SOCKET"])
        ready.settimeout(10)
        for index, argv in enumerate(pre):
            if index == 1 and env.get("MERGE_GATE_TEST_FAIL_SETUP"):
                for cleanup in post:
                    subprocess.run(cleanup, cwd=cwd, env=env, check=True)
                print("injected restore failure", file=sys.stderr)
                return 1
            subprocess.run(argv, cwd=cwd, env=env, check=True, stdout=log, stderr=log)
        # A separate supervisor executes the actual entry point and reaps the
        # checkout after it exits, even when it never returns normally.
        supervise = """
import json, os, socket, subprocess, sys
from pathlib import Path
post, *command = sys.argv[1:]
try:
    result = subprocess.run([sys.executable, *command], check=False)
finally:
    for argv in json.loads(post):
        subprocess.run(argv, check=True)
    if address := os.environ.get("MERGE_GATE_TEST_STOPPED"):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as notice:
            notice.connect(address)
            notice.sendall(str(result.returncode).encode())
"""
        subprocess.Popen(
            [command[0], "-c", supervise, json.dumps(post), *command[1:]],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        assert ready.recv(1024) == b"READY=1"
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
