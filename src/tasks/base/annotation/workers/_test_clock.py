"""Supervisor-side logical clock, enabled only by an explicit test-layout fd.

The control socket is inherited by the supervisor, never by the worker. Both the
real child exit and test clock commands wake the same selector; no parent clock
patch or wall-time delay is needed to exercise the actual timeout/cleanup path.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import socket
import subprocess
from collections.abc import Callable


def wait_for_child(
    child: subprocess.Popen[bytes],
    control_fd: int,
    timeout_seconds: float,
    watch: Callable[[], object],
) -> int:
    elapsed = 0.0
    with (
        socket.socket(fileno=os.dup(control_fd)) as control,
        selectors.DefaultSelector() as events,
    ):
        child_fd = os.pidfd_open(child.pid)
        try:
            events.register(control, selectors.EVENT_READ, "clock")
            events.register(child_fd, selectors.EVENT_READ, "child")
            control.send(
                json.dumps(
                    {"state": "ready", "timeout_seconds": timeout_seconds}
                ).encode()
            )
            while True:
                # A finished child wins the boundary race, as in the real clock.
                code = child.poll()
                if code is not None:
                    return code
                if elapsed >= timeout_seconds:
                    raise subprocess.TimeoutExpired(child.args, timeout_seconds)
                for event, _mask in events.select(timeout=2):
                    if event.data != "clock":
                        continue
                    message = control.recv(4096)
                    if not message:
                        raise ValueError("test supervisor clock disconnected")
                    value = float(json.loads(message)["elapsed_seconds"])
                    if not math.isfinite(value) or value < elapsed:
                        raise ValueError("test clock must advance monotonically")
                    elapsed = value
                    code = child.poll()
                    control.send(
                        json.dumps(
                            {
                                "elapsed_seconds": elapsed,
                                "state": "exited"
                                if code is not None
                                else "timed_out"
                                if elapsed >= timeout_seconds
                                else "running",
                            }
                        ).encode()
                    )
                watch()
        finally:
            os.close(child_fd)
