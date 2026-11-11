"""Detach the queue service from this finite Codex session after all four registrations."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from control import ROOT, now, read, store, write


def main():
    request = read(ROOT / "queue-service-request.json")
    if not request["ready"]:
        raise ValueError("The request is already claimed; inspect the current service")
    state_path = ROOT / "queue-service-state.json"
    if state_path.exists():
        state = read(state_path)
        try:
            os.kill(state["service_pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("An existing service may still own the queue")
    with Path(request["log"]).open("ab", buffering=0) as log:
        process = subprocess.Popen(
            request["argv"],
            cwd=request["cwd"],
            env=dict(os.environ, **request["env"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    record = dict(
        schema="ball_query_control_service_launch.v1",
        launched_at=now(),
        pid=process.pid,
        request=request,
    )
    write(ROOT / "service-launch.json", record)
    store().publish_file(ROOT / "service-launch.json")
    # Finite handshake only. The service owns all children after this script exits.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited before admission: {process.returncode}")
        if state_path.exists() and read(state_path)["service_pid"] == process.pid:
            print(json.dumps(record), flush=True)
            return
        time.sleep(0.05)
    raise TimeoutError("No service admission handshake; inspect before retrying")


if __name__ == "__main__":
    main()
