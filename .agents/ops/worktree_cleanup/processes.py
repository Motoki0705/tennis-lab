"""Identify native agents and stop only pinned process identities and descendants."""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import time
from pathlib import Path
from threading import Event
from typing import Any

from lib.opsproc import process_identity

from .storage import CleanupError


def within(value: str, root: Path) -> bool:
    if not isinstance(value, str):
        raise CleanupError("managed worktree path must be a string")
    return Path(value).is_absolute() and Path(value).is_relative_to(root)


def open_pidfd(pid: int) -> int:
    # The host's Python may have been built without these optional os/signal
    # APIs even though libc and the running Linux kernel provide pidfds.
    call = getattr(ctypes.CDLL(None, use_errno=True), "pidfd_open", None)
    if call is None:
        raise CleanupError("libc pidfd_open is required for native agent stop")
    call.argtypes = [ctypes.c_int, ctypes.c_uint]
    call.restype = ctypes.c_int
    fd = call(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return int(fd)


def signal_pidfd(fd: int, sig: int) -> None:
    call = getattr(ctypes.CDLL(None, use_errno=True), "pidfd_send_signal", None)
    if call is None:
        raise CleanupError("libc pidfd_send_signal is required for native agent stop")
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(fd, sig, None, 0):
        code = ctypes.get_errno()
        if code != errno.ESRCH:  # The pinned process already exited.
            raise OSError(code, "pidfd_send_signal failed")


def agents(root: Path) -> list[dict[str, Any]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            names = [Path(os.fsdecode(arg)).name for arg in argv[:2]]
            if not any(name in {"codex", "claude"} for name in names):
                continue
            cwd = os.readlink(entry / "cwd").removesuffix(" (deleted)")
            if not within(cwd, root):
                continue
            environment = dict(
                item.split(b"=", 1)
                for item in (entry / "environ").read_bytes().split(b"\0")
                if b"=" in item
            )
            sessions = [
                "codex:" + os.fsdecode(environment[key])
                for key in (b"CODEX_SESSION_ID", b"CODEX_THREAD_ID")
                if environment.get(key)
            ]
            if environment.get(b"CLAUDE_CODE_SESSION_ID"):
                sessions.append(
                    "claude:" + os.fsdecode(environment[b"CLAUDE_CODE_SESSION_ID"])
                )
            ident = process_identity(int(entry.name))
            if ident:
                found.append({"identity": ident, "sessions": sessions})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise CleanupError(
                f"cannot inspect local agent process {entry.name}"
            ) from exc
    return found


def run_directories(root: Path) -> set[Path]:
    found = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid() or not within(
                os.readlink(entry / "cwd").removesuffix(" (deleted)"), root
            ):
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
            for value in environment:
                if value.startswith(b"AGENT_RUNTIME_RUN_DIR="):
                    directory = Path(os.fsdecode(value.partition(b"=")[2]))
                    if not directory.is_absolute():
                        raise CleanupError("agent runtime directory is not absolute")
                    found.add(directory)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def stop_tree(ident: dict[str, Any], timeout: float = 5.0) -> None:
    if process_identity(ident["pid"]) != ident:
        return
    parents = {}
    identities = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                raw = (entry / "stat").read_text()
                parents[int(entry.name)] = int(raw[raw.rindex(")") + 2 :].split()[1])
                identities[int(entry.name)] = process_identity(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
    ancestors = {os.getpid()}
    current = os.getpid()
    while (current := parents.get(current, 0)) and current not in ancestors:
        ancestors.add(current)
    if ident["pid"] in ancestors:
        raise CleanupError(
            "cleanup cannot stop its own agent ancestor; run from the main checkout service"
        )
    pids = [ident["pid"]]
    for pid in pids:
        pids.extend(
            child
            for child, parent in parents.items()
            if parent == pid and child not in pids
        )
    pinned = []
    try:
        for pid in reversed(pids):
            observed = process_identity(pid)
            if observed is None:
                continue
            if observed != identities.get(pid) or (
                pid == ident["pid"] and observed != ident
            ):
                raise CleanupError("process identity changed during native agent stop")
            try:
                fd = open_pidfd(pid)
            except ProcessLookupError:
                continue
            if process_identity(pid) != observed:
                os.close(fd)
                continue
            pinned.append((fd, observed))
            signal_pidfd(fd, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while any(process_identity(item["pid"]) == item for _, item in pinned):
            if time.monotonic() >= deadline:
                for fd, item in pinned:
                    if process_identity(item["pid"]) == item:
                        signal_pidfd(fd, signal.SIGKILL)
                break
            Event().wait(0.05)
        deadline = time.monotonic() + timeout
        while any(process_identity(item["pid"]) == item for _, item in pinned):
            if time.monotonic() >= deadline:
                raise CleanupError("agent descendants survived SIGKILL")
            Event().wait(0.05)
    finally:
        for fd, _ in pinned:
            os.close(fd)
