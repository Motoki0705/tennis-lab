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
                # Observe path scope alongside the baseline, before the later
                # agent/registry discovery. A short-lived unrelated process
                # can exit between these two passes on a busy host.
                uid = entry.stat().st_uid
                try:
                    cwd = (
                        os.readlink(entry / "cwd").removesuffix(" (deleted)")
                        if uid == os.getuid()
                        else None
                    )
                except PermissionError:
                    cwd = None  # Lineage still covers known descendants.
                ident = process_identity(int(entry.name))
                if ident is not None and ident["start_ticks"] == fields[19]:
                    found[int(entry.name)] = {
                        "identity": ident,
                        "parent": int(fields[1]),
                        "uid": uid,
                        "cwd": cwd,
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


def scoped_processes(
    root: Path, view: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Find all cwd roots and retain the identities proven outside this scope."""
    found = []
    excluded = set()
    for pid, node in view.items():
        if "uid" in node and "cwd" in node:
            # This evidence belongs to the pinned baseline identity, including
            # after it exits. A later PID owner must not replace its cwd.
            if node["uid"] != os.getuid() or (
                node["cwd"] is not None and not within(node["cwd"], root)
            ):
                excluded.add(identity_key(node["identity"]))
            elif node["cwd"] is not None:
                found.append(node["identity"])
            continue
        entry = Path(f"/proc/{pid}")
        try:
            if entry.stat().st_uid != os.getuid():
                if process_identity(pid) == node["identity"]:
                    excluded.add(identity_key(node["identity"]))
                continue
            cwd = os.readlink(entry / "cwd").removesuffix(" (deleted)")
            if process_identity(pid) == node["identity"]:
                if within(cwd, root):
                    found.append(node["identity"])
                else:
                    excluded.add(identity_key(node["identity"]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # An unrelated same-UID sandbox can hide its cwd. It is not
            # selected by path; known descendants remain covered by lineage.
            continue
    return found, excluded


def expand_tracked(tracked: dict[str, Any], view: dict[int, dict[str, Any]]) -> None:
    changed = True
    while changed:
        changed = False
        parents = {
            item["pid"]
            for item in tracked.values()
            if view.get(item["pid"], {}).get("identity") == item
        }
        for node in view.values():
            key = identity_key(node["identity"])
            if node["parent"] in parents and key not in tracked:
                tracked[key] = node["identity"]
                changed = True


def read_stop_proof(proof: Path | None) -> dict[str, Any] | None:
    saved: dict[str, Any] | None = (
        read_json(proof) if proof is not None and proof.exists() else None
    )
    if saved is None:
        return None
    version = saved.get("schema_version")
    if (
        type(version) is not int
        or version not in {1, 2, 3}
        or set(saved)
        != {"schema_version", "baseline", "ancestors", "tracked"}
        | ({"parents", "capture_complete", "unconfirmed"} if version >= 2 else set())
        or not all(
            isinstance(saved[k], list) for k in ("baseline", "ancestors", "tracked")
        )
    ):
        raise CleanupError("invalid native stop proof")
    if version >= 2 and not isinstance(saved["unconfirmed"], list):
        raise CleanupError("invalid native uncertainty in stop proof")
    for item in (
        *saved["baseline"],
        *saved["ancestors"],
        *saved["tracked"],
        *saved.get("unconfirmed", []),
    ):
        if (
            not isinstance(item, dict)
            or set(item) != {"pid", "boot_id", "start_ticks"}
            or type(item["pid"]) is not int
            or item["pid"] <= 0
            or not isinstance(item["boot_id"], str)
            or not isinstance(item["start_ticks"], str)
        ):
            raise CleanupError("invalid native process identity in stop proof")
    if version >= 2 and (
        type(saved["capture_complete"]) is not bool
        or not isinstance(saved["parents"], dict)
        or set(saved["parents"]) != {identity_key(item) for item in saved["baseline"]}
        or any(type(pid) is not int or pid < 0 for pid in saved["parents"].values())
    ):
        raise CleanupError("invalid native ancestry in stop proof")
    return saved


def unscoped_recovery_evidence(
    view: dict[int, dict[str, Any]],
    seeds: list[dict[str, Any]],
    saved: dict[str, Any] | None,
    root: Path,
) -> dict[str, Any] | None:
    """Rebase a completed v2 discovery that never captured a worktree root.

    V2 recorded every vanished host PID as a possible root, even if it was a
    transient tool of another job. Its ancestors cannot attribute *future*
    unrelated jobs indefinitely. Retain the original proof; require every
    captured uncertain identity (including its captured children) to have ended
    and a fresh empty worktree scope. Known roots and interrupted capture are
    deliberately ineligible for this recovery.
    """
    if (
        saved is None
        or saved["schema_version"] != 2
        or not saved["capture_complete"]
        or saved["tracked"]
        or seeds
        or not saved["unconfirmed"]
        or any(
            view.get(item["pid"], {}).get("identity") == item
            for item in saved["unconfirmed"]
        )
    ):
        return None
    return {
        "worktree": str(root),
        "reason": "legacy_unscoped_roots_ended",
        "captured_roots": 0,
        "ended_unconfirmed": saved["unconfirmed"],
    }


def recover_unscoped_stop(
    view: dict[int, dict[str, Any]],
    seeds: list[dict[str, Any]],
    proof: Path,
    root: Path,
) -> dict[str, Any] | None:
    saved = read_stop_proof(proof)
    evidence = unscoped_recovery_evidence(view, seeds, saved, root)
    if evidence is None:
        return None
    archive = proof.with_suffix(".unscoped-v2.json")
    original = {"proof": saved, "recovery": evidence}
    if archive.exists() and read_json(archive) != original:
        raise CleanupError("conflicting legacy unscoped stop recovery")
    write_json(archive, original)
    write_json(proof, prepare_stop(view, [], None, capture_completed=True))
    return evidence


def uncertain_processes(
    view: dict[int, dict[str, Any]],
    tracked: dict[str, Any],
    baseline: dict[str, Any],
    outside: dict[str, Any],
    unconfirmed: dict[str, Any],
) -> list[dict[str, Any]]:
    """Classify unresolved provenance without signals or proof writes."""
    result = []
    for node in view.values():
        key = identity_key(node["identity"])
        if key in tracked:
            continue
        if key in unconfirmed:
            result.append(node["identity"])
            continue
        if key in baseline:
            continue
        parent, seen, unattributed = node["parent"], set(), True
        while parent in view and parent not in seen:
            seen.add(parent)
            ancestor = view[parent]
            ancestor_key = identity_key(ancestor["identity"])
            if ancestor_key in outside or ancestor_key in unconfirmed:
                break
            if ancestor_key in baseline and ancestor_key not in tracked:
                unattributed = False
                break
            parent = ancestor["parent"]
        if unattributed:
            result.append(node["identity"])
    return result


def prepare_stop(
    view: dict[int, dict[str, Any]],
    seeds: list[dict[str, Any]],
    proof: Path | None,
    *,
    capture_completed: bool = False,
    uncertain_roots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist the first baseline and all roots before any managed stop action."""
    saved = read_stop_proof(proof)
    baseline = (
        {identity_key(item): item for item in saved["baseline"]}
        if saved
        else {
            identity_key(node["identity"]): node["identity"] for node in view.values()
        }
    )
    lineage = (
        dict(saved.get("parents", {}))
        if saved
        else {identity_key(node["identity"]): node["parent"] for node in view.values()}
    )
    # Version-1 proofs have no ancestry map. Never infer a vanished root's
    # lineage from a later PID owner; make all new provenance uncertain instead.
    if saved and "parents" not in saved:
        lineage = {key: 0 for key in baseline}
    old_view = {
        item["pid"]: {"identity": item, "parent": lineage[key]}
        for key, item in baseline.items()
    }
    tracked = {
        identity_key(item): item
        for item in [*(saved["tracked"] if saved else []), *seeds]
    }
    unconfirmed = {
        identity_key(item): item
        for item in [
            *(saved.get("unconfirmed", []) if saved else []),
            *(uncertain_roots or []),
        ]
    }
    outside = {
        identity_key(item): item
        for item in (saved["ancestors"] if saved and not capture_completed else [])
    }
    for item in [*seeds, *(uncertain_roots or [])]:
        node = old_view.get(item["pid"])
        if (
            node is None
            or node["identity"] != item
            or (saved and "parents" not in saved)
        ):
            outside.update(
                baseline
            )  # A root disappeared before its lineage was captured.
            continue
        parent, seen = node["parent"], set()
        while parent in old_view and parent not in seen:
            seen.add(parent)
            ancestor = old_view[parent]
            outside[identity_key(ancestor["identity"])] = ancestor["identity"]
            parent = ancestor["parent"]
        if parent:
            # A missing ancestor/cycle cannot prove which existing subreaper
            # could inherit children. Do not infer an unrelated branch.
            outside.update(baseline)
    expand_tracked(tracked, old_view)
    expand_tracked(tracked, view)
    # A process can disappear before its cwd is read. Baseline children of
    # that unknown root are not proven unrelated merely because they predate
    # discovery. Preserve their uncertainty without signalling them by guess.
    expand_tracked(unconfirmed, old_view)
    expand_tracked(unconfirmed, view)
    result = {
        # Keep unresolved legacy uncertainty eligible for the explicit recovery
        # above, until all identities have ended. Fresh captures use v3.
        "schema_version": 2
        if saved and saved["schema_version"] == 2 and not tracked
        else 3,
        "baseline": list(baseline.values()),
        "ancestors": list(outside.values()),
        "tracked": list(tracked.values()),
        "unconfirmed": list(unconfirmed.values()),
        "parents": lineage,
        "capture_complete": capture_completed
        or bool(saved and saved.get("capture_complete", True)),
    }
    if proof is not None:
        write_json(proof, result)
    ancestors = {os.getpid()}
    current = os.getpid()
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
    return result


def begin_stop(view: dict[int, dict[str, Any]], proof: Path) -> bool:
    """Journal before discovery. An interrupted discovery cannot certify roots."""
    if proof.exists():
        saved = read_stop_proof(proof)
        if saved is not None and not saved.get("capture_complete", True):
            raise CleanupError(
                "native discovery was interrupted; original roots cannot be certified"
            )
        return False
    data = prepare_stop(view, [], None)
    # Only this invocation, after finishing discovery, may narrow this set.
    # A resumed attempt cannot certify which original roots it missed.
    data["ancestors"] = list(data["baseline"])
    write_json(proof, data)
    return True


def stop_tree(
    ident: dict[str, Any] | None, timeout: float = 5.0, *, proof: Path | None = None
) -> None:
    """Freeze a closed forest before KILL; replay the preparation's provenance."""
    view = snapshot()
    saved = read_stop_proof(proof)
    if saved is not None and not saved.get("capture_complete", True):
        raise CleanupError(
            "native discovery was interrupted; original roots cannot be certified"
        )
    if saved is None and (
        ident is None or view.get(ident["pid"], {}).get("identity") != ident
    ):
        return  # Raw pin helper only; Stopper always supplies its durable proof.
    saved = prepare_stop(
        view, [] if ident is None else [ident], proof, capture_completed=saved is None
    )
    tracked = {identity_key(item): item for item in saved["tracked"]}
    baseline = {identity_key(item): item for item in saved["baseline"]}
    outside = {identity_key(item): item for item in saved["ancestors"]}
    unconfirmed = {identity_key(item): item for item in saved["unconfirmed"]}

    def expand(current_view: dict[int, dict[str, Any]]) -> None:
        expand_tracked(tracked, current_view)
        expand_tracked(unconfirmed, current_view)

    def save() -> None:
        if proof is not None:
            write_json(
                proof,
                {
                    "schema_version": saved["schema_version"],
                    "parents": saved["parents"],
                    "capture_complete": saved["capture_complete"],
                    "baseline": list(baseline.values()),
                    "ancestors": list(outside.values()),
                    "tracked": list(tracked.values()),
                    "unconfirmed": list(unconfirmed.values()),
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
            unknown = uncertain_processes(view, tracked, baseline, outside, unconfirmed)
            if not live and not unknown:
                break
            if time.monotonic() >= deadline:
                raise CleanupError(
                    "native descendants survived or new orphan provenance is unconfirmed: "
                    f"live={live}; uncertain={unknown}"
                )
            Event().wait(0.05)
        if proof is not None and proof.exists():
            proof.unlink()
            sync_directory(proof.parent)
    finally:
        for fd in pinned.values():
            os.close(fd)
