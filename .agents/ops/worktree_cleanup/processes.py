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

from .storage import CleanupError, read_json, sync_directory, write_json


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


def snapshot() -> dict[int, dict[str, Any]]:
    found = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                raw = (entry / "stat").read_text()
                fields = raw[raw.rindex(")") + 2 :].split()
                ident = process_identity(int(entry.name))
                if ident is not None and ident["start_ticks"] == fields[19]:
                    found[int(entry.name)] = {
                        "identity": ident,
                        "parent": int(fields[1]),
                    }
            except (FileNotFoundError, ProcessLookupError):
                continue
    return found


def threads_stopped(ident: dict[str, Any]) -> bool:
    if process_identity(ident["pid"]) != ident:
        return True
    try:
        tasks = list(Path(f"/proc/{ident['pid']}/task").iterdir())
        for task in tasks:
            try:
                raw = (task / "stat").read_text()
            except FileNotFoundError:
                continue
            if raw[raw.rindex(")") + 2 :].split()[0] not in {"T", "t", "Z"}:
                return False
        return bool(tasks) and process_identity(ident["pid"]) == ident
    except FileNotFoundError:
        return True


def identity_key(ident: dict[str, Any]) -> str:
    return f"{ident['pid']}:{ident['boot_id']}:{ident['start_ticks']}"


def stop_tree(
    ident: dict[str, Any] | None, timeout: float = 5.0, *, proof: Path | None = None
) -> None:
    """Freeze a closed forest before KILL; retain identities across reparenting.

    Unknown new processes below an outside ancestor can be newly orphaned
    descendants. Never kill those by guess or certify completion while they
    survive. The durable proof retains this uncertainty for a later invocation.
    """
    view = snapshot()
    saved = read_json(proof) if proof is not None and proof.exists() else None
    if saved is not None:
        if (
            set(saved) != {"schema_version", "baseline", "ancestors", "tracked"}
            or type(saved["schema_version"]) is not int
            or saved["schema_version"] != 1
            or not all(
                isinstance(saved[k], list) for k in ("baseline", "ancestors", "tracked")
            )
        ):
            raise CleanupError("invalid native stop proof")
        for item in (*saved["baseline"], *saved["ancestors"], *saved["tracked"]):
            if (
                not isinstance(item, dict)
                or set(item) != {"pid", "boot_id", "start_ticks"}
                or type(item["pid"]) is not int
                or item["pid"] <= 0
                or not isinstance(item["boot_id"], str)
                or not isinstance(item["start_ticks"], str)
            ):
                raise CleanupError("invalid native process identity in stop proof")
    seeds = [] if ident is None else [ident]
    seeds += [] if saved is None else saved["tracked"]
    tracked = {
        identity_key(item): item
        for item in seeds
        if view.get(item["pid"], {}).get("identity") == item
    }
    if not tracked and saved is None:
        return
    ancestors = {os.getpid()}
    current = os.getpid()
    while (
        current := view.get(current, {}).get("parent", 0)
    ) and current not in ancestors:
        ancestors.add(current)
    if any(item["pid"] in ancestors for item in tracked.values()):
        raise CleanupError(
            "cleanup cannot stop its own agent ancestor; run from the main checkout service"
        )
    baseline = {
        identity_key(item): item
        for item in (
            saved["baseline"] if saved else [v["identity"] for v in view.values()]
        )
    }
    outside = {
        identity_key(item): item for item in (saved["ancestors"] if saved else [])
    }
    if saved is None:
        for item in tracked.values():
            parent = view[item["pid"]]["parent"]
            seen = set()
            while parent in view and parent not in seen:
                seen.add(parent)
                ancestor = view[parent]
                outside[identity_key(ancestor["identity"])] = ancestor["identity"]
                parent = ancestor["parent"]

    def expand(current_view: dict[int, dict[str, Any]]) -> None:
        changed = True
        while changed:
            changed = False
            parents = {
                item["pid"]
                for item in tracked.values()
                if current_view.get(item["pid"], {}).get("identity") == item
            }
            for node in current_view.values():
                key = identity_key(node["identity"])
                if node["parent"] in parents and key not in tracked:
                    tracked[key] = node["identity"]
                    changed = True

    def uncertain(current_view: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for node in current_view.values():
            key = identity_key(node["identity"])
            if key in baseline or key in tracked:
                continue
            parent, seen, unconfirmed = node["parent"], set(), True
            while parent in current_view and parent not in seen:
                seen.add(parent)
                ancestor = current_view[parent]
                ancestor_key = identity_key(ancestor["identity"])
                if ancestor_key in outside:
                    break
                if ancestor_key in baseline and ancestor_key not in tracked:
                    unconfirmed = False
                    break  # A different, already existing process owns this branch.
                parent = ancestor["parent"]
            if unconfirmed:
                result.append(node["identity"])
        return result

    def save() -> None:
        if proof is not None:
            write_json(
                proof,
                {
                    "schema_version": 1,
                    "baseline": list(baseline.values()),
                    "ancestors": list(outside.values()),
                    "tracked": list(tracked.values()),
                },
            )

    expand(view)
    save()
    pinned: dict[str, int] = {}
    try:
        deadline = time.monotonic() + timeout
        while True:
            view = snapshot()
            expand(view)
            save()  # Persist newly found children before signalling their parents.
            for key, item in tracked.items():
                if key in pinned or view.get(item["pid"], {}).get("identity") != item:
                    continue
                try:
                    fd = open_pidfd(item["pid"])
                except ProcessLookupError:
                    continue
                if process_identity(item["pid"]) != item:
                    os.close(fd)
                    continue
                pinned[key] = fd
                signal_pidfd(fd, signal.SIGSTOP)
            view = snapshot()
            old_count = len(tracked)
            expand(view)
            live = [
                item
                for item in tracked.values()
                if view.get(item["pid"], {}).get("identity") == item
            ]
            if len(tracked) == old_count and all(
                threads_stopped(item) for item in live
            ):
                # A second snapshot after every thread stopped closes the last
                # fork-in-flight window. No TERM handler or SIGCONT may run.
                final_view = snapshot()
                expand(final_view)
                if len(tracked) == old_count and all(
                    threads_stopped(item) for item in live
                ):
                    break
            if time.monotonic() >= deadline:
                raise CleanupError("native process forest did not become fully stopped")
            Event().wait(0.05)
        save()
        for key, fd in pinned.items():
            if process_identity(tracked[key]["pid"]) == tracked[key]:
                signal_pidfd(fd, signal.SIGKILL)
        deadline = time.monotonic() + timeout
        while True:
            view = snapshot()
            expand(view)
            save()
            live = [
                item
                for item in tracked.values()
                if view.get(item["pid"], {}).get("identity") == item
            ]
            unknown = uncertain(view)
            if not live and not unknown:
                break
            if time.monotonic() >= deadline:
                raise CleanupError(
                    "native descendants survived or new orphan provenance is unconfirmed"
                )
            Event().wait(0.05)
        if proof is not None and proof.exists():
            proof.unlink()
            sync_directory(proof.parent)
    finally:
        for fd in pinned.values():
            os.close(fd)
