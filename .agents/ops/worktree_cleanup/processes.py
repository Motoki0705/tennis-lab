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


def agent_metadata(ident: dict[str, Any]) -> dict[str, Any] | None:
    """Read sessions only from a pinned Codex/Claude, never inherited tool env."""
    entry = Path(f"/proc/{ident['pid']}")
    try:
        if process_identity(ident["pid"]) != ident:
            return None
        argv = (entry / "cmdline").read_bytes().split(b"\0")
        names = [Path(os.fsdecode(arg)).name for arg in argv[:2]]
        if not any(name in {"codex", "claude"} for name in names):
            return None
        try:
            environment = dict(
                item.split(b"=", 1)
                for item in (entry / "environ").read_bytes().split(b"\0")
                if b"=" in item
            )
        except PermissionError:
            environment = {}
        sessions = [
            "codex:" + os.fsdecode(environment[key])
            for key in (b"CODEX_SESSION_ID", b"CODEX_THREAD_ID")
            if environment.get(key)
        ]
        if environment.get(b"CLAUDE_CODE_SESSION_ID"):
            sessions.append(
                "claude:" + os.fsdecode(environment[b"CLAUDE_CODE_SESSION_ID"])
            )
        if process_identity(ident["pid"]) == ident:
            return {"identity": ident, "sessions": sessions}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def agents(root: Path) -> list[dict[str, Any]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            cwd = os.readlink(entry / "cwd").removesuffix(" (deleted)")
            if not within(cwd, root):
                continue
            ident = process_identity(int(entry.name))
            info = agent_metadata(ident) if ident else None
            if info is not None:
                found.append(info)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # Reference inspection separately classifies opaque related PIDs.
            continue
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
    found: dict[int, dict[str, Any]] = {}
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    # Capture ancestry before slower scope probes. Never make retention of an
    # observed stat depend on a second read of a process that may already exit.
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                raw = (entry / "stat").read_text()
                fields = raw[raw.rindex(")") + 2 :].split()
                if fields[0] in {"Z", "X"}:
                    continue  # Already dead before this snapshot observed it.
                pid = int(entry.name)
                found[pid] = {
                    "identity": {
                        "pid": pid,
                        "boot_id": boot,
                        "start_ticks": fields[19],
                    },
                    "parent": int(fields[1]),
                    "uid": None,
                    "cwd": None,
                    "identity_lost": False,
                    "scope_uncertain": False,
                }
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                # Reference scans report these as uninspectable and apply
                # the explicit related/unrelated policy; do not reject the host.
                continue
    for pid, node in found.items():
        entry = Path(f"/proc/{pid}")
        try:
            node["uid"] = entry.stat().st_uid
            if node["uid"] == os.getuid():
                node["cwd"] = os.readlink(entry / "cwd").removesuffix(" (deleted)")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass  # Keep the original identity/ancestry and any observed cwd.
        try:
            current = process_identity(pid)
        except PermissionError:
            # Losing inspection rights must not discard earlier observations
            # or let a retry certify their children as an unrelated baseline.
            current = None
            node["scope_uncertain"] = True
        if current != node["identity"]:
            node["identity_lost"] = True
            if current is not None:
                # Scope could belong to the replacement; its identity must not
                # replace the original stat or authorize signalling the new PID.
                node["scope_uncertain"] = True
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


def valid_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"pid", "boot_id", "start_ticks"}
        and type(value["pid"]) is int
        and value["pid"] > 0
        and isinstance(value["boot_id"], str)
        and bool(value["boot_id"])
        and isinstance(value["start_ticks"], str)
        and value["start_ticks"].isdecimal()
    )


def scoped_processes(
    root: Path, view: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Retain pinned identities whose cwd was observed in this worktree."""
    found = []
    for pid, node in view.items():
        if "uid" in node and "cwd" in node:
            if node.get("scope_uncertain"):
                continue  # Retain observations without trusting another generation's scope.
            # This evidence belongs to the observed identity, including
            # after it exits. A later PID owner must not replace its cwd.
            if (
                node["uid"] == os.getuid()
                and node["cwd"] is not None
                and within(node["cwd"], root)
            ):
                found.append(node["identity"])
            continue
        entry = Path(f"/proc/{pid}")
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            cwd = os.readlink(entry / "cwd").removesuffix(" (deleted)")
            if process_identity(pid) == node["identity"] and within(cwd, root):
                found.append(node["identity"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # An unrelated same-UID sandbox can hide its cwd. It is not
            # selected by path; known descendants remain covered by lineage.
            continue
    return found


def expand_tracked(tracked: dict[str, Any], view: dict[int, dict[str, Any]]) -> None:
    changed = True
    while changed:
        changed = False
        parents = {
            item["pid"]
            for item in tracked.values()
            if view.get(item["pid"], {}).get("identity") == item
            and not view[item["pid"]].get("scope_uncertain")
        }
        for node in view.values():
            key = identity_key(node["identity"])
            if (
                node["parent"] in parents
                and key not in tracked
                and int(node["identity"]["start_ticks"])
                >= int(view[node["parent"]]["identity"]["start_ticks"])
            ):
                tracked[key] = node["identity"]
                changed = True


def read_stop_proof(proof: Path | None) -> dict[str, Any] | None:
    """Read only the current journal of observed identities, never orphan guesses."""
    saved: dict[str, Any] | None = (
        read_json(proof) if proof is not None and proof.exists() else None
    )
    if saved is None:
        return None
    if (
        set(saved) != {"schema_version", "tracked"}
        or type(saved["schema_version"]) is not int
        or saved["schema_version"] != 4
        or not isinstance(saved["tracked"], list)
    ):
        raise CleanupError("invalid observed native stop journal")
    for item in saved["tracked"]:
        if not valid_identity(item):
            raise CleanupError("invalid observed native process identity")
    return saved


def prepare_stop(
    view: dict[int, dict[str, Any]], seeds: list[dict[str, Any]], proof: Path | None
) -> dict[str, Any]:
    """Journal observed roots and descendants before managed jobs can exit."""
    saved = read_stop_proof(proof)
    tracked = {
        identity_key(item): item
        for item in [*(saved["tracked"] if saved else []), *seeds]
    }
    expand_tracked(tracked, view)
    ancestors, current = {os.getpid()}, os.getpid()
    while (
        current := view.get(current, {}).get("parent", 0)
    ) and current not in ancestors:
        ancestors.add(current)
    if any(
        item["pid"] in ancestors and view.get(item["pid"], {}).get("identity") == item
        for item in tracked.values()
    ):
        raise CleanupError(
            "cleanup cannot stop its own agent ancestor; run from the main checkout service"
        )
    result = {"schema_version": 4, "tracked": list(tracked.values())}
    if proof is not None:
        write_json(proof, result)
    return result


def stop_tree(
    ident: dict[str, Any] | None, timeout: float = 5.0, *, proof: Path | None = None
) -> None:
    """Stop pinned observed identities. Unobserved orphan guesses are not a gate."""
    view = snapshot()
    if proof is None and (ident is None or process_identity(ident["pid"]) != ident):
        return
    saved = prepare_stop(view, [] if ident is None else [ident], proof)
    tracked = {identity_key(item): item for item in saved["tracked"]}

    def save() -> None:
        if proof is not None:
            write_json(proof, {"schema_version": 4, "tracked": list(tracked.values())})

    def live() -> list[dict[str, Any]]:
        try:
            return [
                item
                for item in tracked.values()
                if process_identity(item["pid"]) == item
            ]
        except PermissionError as exc:
            raise CleanupError(
                "cannot confirm an observed native process ended"
            ) from exc

    pinned: dict[str, int] = {}
    try:
        deadline = time.monotonic() + timeout
        while True:
            view = snapshot()
            expand_tracked(tracked, view)
            save()
            for key, item in list(tracked.items()):
                if key in pinned or process_identity(item["pid"]) != item:
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
            old_count = len(tracked)
            expand_tracked(tracked, snapshot())
            if len(tracked) == old_count and all(
                threads_stopped(item) for item in live()
            ):
                # Capture one more view after every observed thread is stopped.
                expand_tracked(tracked, snapshot())
                if len(tracked) == old_count and all(
                    threads_stopped(item) for item in live()
                ):
                    break
            if time.monotonic() >= deadline:
                raise CleanupError(
                    "observed native process forest did not become fully stopped"
                )
            Event().wait(0.05)
        save()
        for key, fd in pinned.items():
            if process_identity(tracked[key]["pid"]) == tracked[key]:
                signal_pidfd(fd, signal.SIGKILL)
        deadline = time.monotonic() + timeout
        while True:
            expand_tracked(tracked, snapshot())
            save()
            remaining = live()
            if not remaining:
                break
            if time.monotonic() >= deadline:
                raise CleanupError(
                    f"observed native processes survived: live={remaining}"
                )
            Event().wait(0.05)
        if proof is not None and proof.exists():
            proof.unlink()
            sync_directory(proof.parent)
    finally:
        for fd in pinned.values():
            os.close(fd)
