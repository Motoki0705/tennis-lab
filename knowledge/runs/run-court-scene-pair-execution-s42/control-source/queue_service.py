"""Run the shared queue outside the finite agent supervisor, reaping orphans.

Start this empty service from the owning parent; the Codex launch operator still
owns enqueueing training and checking its first durable checkpoint. Never invoke
this service inside that operator's agent_exec process tree.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress


def main() -> int:
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("Provide the foreground shared-queue serve command")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot own orphan reaping")
    signals = {signal.SIGCHLD, signal.SIGTERM, signal.SIGINT}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    worker = subprocess.Popen(
        command,
        start_new_session=True,
        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous),
    )
    print(
        json.dumps(
            dict(event="queue_service_started", pid=os.getpid(), worker_pid=worker.pid)
        ),
        flush=True,
    )
    result = None
    while True:
        children = True
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                children = False
                break
            if pid == 0:
                break
            code = os.waitstatus_to_exitcode(status)
            if pid == worker.pid:
                result = code
                worker.returncode = code
            print(
                json.dumps(
                    dict(
                        event="child_reaped",
                        pid=pid,
                        role="worker" if pid == worker.pid else "adopted",
                        exit_code=code,
                    )
                ),
                flush=True,
            )
        if result is not None and not children:
            return result
        event = signal.sigwait(signals)
        if event in {signal.SIGTERM, signal.SIGINT} and result is None:
            # SIGCHLD may already be pending; the next drain records the exit.
            with suppress(ProcessLookupError):
                worker.send_signal(event)


if __name__ == "__main__":
    raise SystemExit(main())
