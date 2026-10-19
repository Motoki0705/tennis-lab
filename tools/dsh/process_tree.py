"""Linux ownership for dsh tools that create their own process groups."""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    group: int
    started: str
    state: str


def process_info(pid: int) -> Process | None:
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(errors="replace")
            .rsplit(")", 1)[1]
            .split()
        )
        return Process(pid, int(fields[1]), int(fields[2]), fields[19], fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


class ProcessTree:
    """Own descendants through reparenting, including setsid/detached tools.

    The runner executes one subprocess interval at a time. Becoming a child
    subreaper makes orphaned tool descendants our direct children, so leader
    exit cannot hide them from a /proc tree walk or prevent their reaping.
    """

    def __init__(self) -> None:
        if not Path("/proc/self/stat").is_file():
            raise OSError("The headless runner requires Linux /proc supervision.")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.previous = ctypes.c_int()
        if (
            self.libc.prctl(37, ctypes.byref(self.previous), 0, 0, 0) != 0
            or self.libc.prctl(36, 1, 0, 0, 0) != 0
        ):
            raise OSError("Cannot enable child subreaper supervision.")

    def members(self) -> list[Process]:
        table = {}
        for path in Path("/proc").iterdir():
            if (
                path.name.isdigit()
                and (process := process_info(int(path.name))) is not None
            ):
                table[process.pid] = process
        owned = {os.getpid()}
        while True:
            added = {
                process.pid for process in table.values() if process.parent in owned
            } - owned
            if not added:
                break
            owned.update(added)
        return [table[pid] for pid in owned if pid != os.getpid()]

    def signal_members(self, members: list[Process], number: signal.Signals) -> None:
        groups = set()
        for process in members:
            current = process_info(process.pid)
            if (
                current is None
                or current.started != process.started
                or current.state == "Z"
            ):
                continue
            # The runner's own group can contain its orchestrator. Never signal
            # that group, even if a tool deliberately joins it.
            if current.group != os.getpgrp() and current.group not in groups:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(current.group, number)
                groups.add(current.group)
            with contextlib.suppress(ProcessLookupError):
                os.kill(current.pid, number)

    def stop(self, child: subprocess.Popen[bytes]) -> None:
        self.signal_members(self.members(), signal.SIGTERM)
        time.sleep(0.2)
        deadline = time.monotonic() + 5
        while True:
            child.poll()
            members = self.members()
            alive = [process for process in members if process.state != "Z"]
            if not alive:
                break
            self.signal_members(alive, signal.SIGKILL)
            if time.monotonic() >= deadline:
                raise OSError("dsh descendants did not stop after SIGKILL.")
            time.sleep(0.02)
        child.wait(timeout=5)
        for process in self.members():
            if process.parent == os.getpid() and process.pid != child.pid:
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(process.pid, os.WNOHANG)

    def close(self) -> None:
        if self.libc.prctl(36, self.previous.value, 0, 0, 0) != 0:
            raise OSError("Cannot restore child subreaper supervision.")
