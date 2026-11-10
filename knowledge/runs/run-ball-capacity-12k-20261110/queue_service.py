"""External subreaper and event-driven terminal notifications for this queue.

Only the owning parent starts this service, outside the finite Codex worker.
The common queue retains its capacity lock and process-group ownership.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import select
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

from campaign import DRIVE, QUEUE, RELATIVE, now, read, sha, write


def ancestors(pid):
    result = []
    while pid > 1:
        path = Path(f"/proc/{pid}")
        raw = (path / "stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        result.append(
            dict(
                pid=pid,
                ppid=int(fields[1]),
                command=(path / "comm").read_text().strip(),
            )
        )
        pid = int(fields[1])
    return result


def notify(root, name, value):
    local = root / "notifications" / name
    write(local, value)
    destination = DRIVE / RELATIVE / "notifications" / name
    write(destination, value)
    if sha(local) != sha(destination):
        raise ValueError("Notification Drive readback differs")


def serve(root, command):
    if (root / "FAILED.json").exists():
        raise ValueError("Explicit failure recovery is required before service restart")
    registration = read(root / "queue-registration.json")
    if not registration["complete"] or len(registration["jobs"]) != 9:
        raise ValueError("All nine jobs must be saved before service admission")
    lineage = ancestors(os.getpid())
    if any(
        x["command"] == "agent_runtime.p"
        or "agent_runtime" in Path(f"/proc/{x['pid']}/cmdline").read_text()
        for x in lineage
    ):
        raise ValueError("Service cannot be a descendant of the finite agent_runtime")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot become a child subreaper")
    inotify = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if inotify < 0:
        raise OSError(ctypes.get_errno(), "Cannot open terminal-state event watch")
    for state in ("failed", "cancelled", "done"):
        directory = QUEUE / state
        directory.mkdir(parents=True, exist_ok=True)
        if (
            libc.inotify_add_watch(
                inotify, os.fsencode(directory), 0x00000008 | 0x00000080
            )
            < 0
        ):
            raise OSError(ctypes.get_errno(), "Cannot watch " + str(directory))
    reader, writer = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    signal.set_wakeup_fd(writer)
    for number in (signal.SIGCHLD, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: None)
    worker = subprocess.Popen(command, start_new_session=True)
    state = dict(
        schema="ball_capacity_queue_service.v1",
        state="running",
        service_pid=os.getpid(),
        worker_pid=worker.pid,
        started_at=now(),
        cwd=str(Path.cwd()),
        argv=command,
        ancestors=lineage,
        notification_mode="inotify terminal events; no periodic progress polling",
    )
    write(root / "queue-service-state.json", state)
    request = read(root / "queue-service-request.json")
    request.update(ready=False, claimed_at=now(), service_pid=os.getpid())
    write(root / "queue-service-request.json", request)
    print(json.dumps(state), flush=True)
    seen, result = set(), None
    while True:
        for job in registration["jobs"]:
            for terminal in ("failed", "cancelled", "done"):
                key = (job["queue_id"], terminal)
                if key in seen or not (QUEUE / terminal / job["queue_file"]).exists():
                    continue
                seen.add(key)
                event = dict(
                    event="queue_" + terminal,
                    at=now(),
                    model=job["model"],
                    queue_id=job["queue_id"],
                    queue_log=str(QUEUE / "logs" / (job["queue_id"] + ".log")),
                )
                if terminal != "done":
                    (QUEUE / "stop").touch()
                    if not (root / "FAILED.json").exists():
                        write(
                            root / "FAILED.json",
                            dict(**event, successors_held=True, automatic_retry=False),
                        )
                try:
                    notify(root, f"{terminal}-{job['queue_id']}.json", event)
                except Exception as error:
                    (QUEUE / "stop").touch()
                    write(
                        root / "FAILED.json",
                        dict(**event, notification_error=repr(error)),
                    )
                    with suppress(ProcessLookupError):
                        worker.send_signal(signal.SIGTERM)
                print(json.dumps(event), flush=True)
        if sum(terminal == "done" for _, terminal in seen) == 9:
            (QUEUE / "stop").touch()
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
                json.dumps(dict(event="child_reaped", pid=pid, exit_code=code)),
                flush=True,
            )
        if result is not None and not children:
            state.update(
                state="failed" if (root / "FAILED.json").exists() else "stopped",
                ended_at=now(),
                exit_code=result,
            )
            write(root / "queue-service-state.json", state)
            try:
                notify(root, "service-ended.json", state)
            except Exception as error:
                state["notification_error"] = repr(error)
                write(root / "queue-service-state.json", state)
                return 1
            return result or (1 if (root / "FAILED.json").exists() else 0)
        ready, _, _ = select.select([reader, inotify], [], [])
        if inotify in ready:
            os.read(inotify, 65536)
        if reader in ready:
            numbers = os.read(reader, 65536)
            for number in numbers:
                if number in (signal.SIGTERM, signal.SIGINT) and result is None:
                    with suppress(ProcessLookupError):
                        worker.send_signal(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("The foreground common queue serve command is required")
    with (args.root / "queue-service.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return serve(args.root, command)


if __name__ == "__main__":
    raise SystemExit(main())
