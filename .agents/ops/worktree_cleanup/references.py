"""Inspect worktree path references; opaque unrelated processes are diagnostics."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from . import processes


def mapped_paths(text: str) -> list[str]:
    paths: list[str] = []
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 5:
            raise ValueError("malformed /proc maps record")
        if len(fields) == 6 and fields[5].startswith("/"):
            # Linux quotes newlines in pathname fields; retain both spellings
            # since a literal backslash-octal spelling is ambiguous in maps.
            paths.extend(
                (
                    fields[5],
                    re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[5]),
                )
            )
    return paths


def _inspect(node: dict[str, Any], root: Path, proc: Path) -> dict[str, Any] | None:
    ident = node["identity"]
    entry = proc / str(ident["pid"])
    matches: list[dict[str, str]] = []
    unreadable: list[dict[str, Any]] = []
    try:
        comm = (entry / "comm").read_text(errors="surrogateescape").strip()
    except OSError:
        comm = "?"

    def failure(item: str, error: Exception) -> None:
        unreadable.append(
            {
                "item": item,
                "error": type(error).__name__,
                "errno": getattr(error, "errno", None),
            }
        )

    # Pin the generation before reading any paths, independently of an older
    # snapshot. A PID replacement must never inherit the old process's label.
    parent, uid = node["parent"], node.get("uid")
    try:
        raw = (entry / "stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        if fields[0] in {"Z", "X"}:
            return None
        ident = {**ident, "start_ticks": fields[19]}
        parent, uid = int(fields[1]), entry.stat().st_uid
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as error:
        failure("stat", error)

    def reference(item: str, value: str) -> None:
        value = value.removesuffix(" (deleted)")
        if value.startswith("/") and processes.within(os.path.normpath(value), root):
            match = {"item": item, "path": value}
            if match not in matches:
                matches.append(match)

    for item in ("cwd", "root", "exe"):
        try:
            value = os.readlink(entry / item)
            if not value.startswith("/"):
                raise ValueError("unresolved process path")
            reference(item, value)
        except (OSError, ValueError) as error:
            failure(item, error)
    try:
        for descriptor in (entry / "fd").iterdir():
            try:
                reference("fd/" + descriptor.name, os.readlink(descriptor))
            except (FileNotFoundError, ProcessLookupError):
                continue  # A descriptor closed during enumeration.
            except OSError as error:
                failure("fd/" + descriptor.name, error)
    except OSError as error:
        failure("fd", error)
    try:
        for value in mapped_paths((entry / "maps").read_text(errors="surrogateescape")):
            reference("maps", value)
    except (OSError, ValueError) as error:
        try:
            for mapping in (entry / "map_files").iterdir():
                try:
                    reference("map_files/" + mapping.name, os.readlink(mapping))
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except OSError as alternative:
            failure("maps", error)
            failure("map_files", alternative)
    try:
        raw = (entry / "stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        if fields[0] in {"Z", "X"}:
            return None
        if fields[19] != ident["start_ticks"]:
            # A path was observed but its final owner is ambiguous. Preserve
            # the positive observation so a fresh scan must resolve it; pidfd
            # stop independently checks the generation before sending signals.
            failure(
                "identity", ValueError("process generation changed during inspection")
            )
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as error:
        failure("stat", error)
    return {
        "identity": ident,
        "parent": parent,
        "uid": uid,
        "comm": comm,
        "references": matches,
        "unreadable": unreadable,
    }


def scan(
    root: Path,
    *,
    related: list[dict[str, Any]] | None = None,
    managed_pids: set[int] | None = None,
    view: dict[int, dict[str, Any]] | None = None,
    proc: Path = Path("/proc"),
) -> dict[str, Any]:
    """Read-only reference gate, bounded to same-UID and observed related PIDs."""
    automatic_view = view is None
    view = processes.snapshot() if view is None else view
    managed_pids = managed_pids or set()
    known = {processes.identity_key(item): item for item in related or []}
    missing = []
    if automatic_view:
        # A process may be new, or its stat was inaccessible in snapshot.
        # Keep inaccessible entries visible without manufacturing identities.
        for entry in proc.iterdir():
            if not entry.name.isdecimal() or int(entry.name) in view:
                continue
            pid = int(entry.name)
            associated = pid in managed_pids or any(
                item["pid"] == pid for item in known.values()
            )
            try:
                uid = entry.stat().st_uid
                if uid != os.getuid() and not associated:
                    continue
                raw = (entry / "stat").read_text()
                fields = raw[raw.rindex(")") + 2 :].split()
                if fields[0] in {"Z", "X"}:
                    continue
                boot = (proc / "sys/kernel/random/boot_id").read_text().strip()
                view[pid] = {
                    "identity": {
                        "pid": pid,
                        "boot_id": boot,
                        "start_ticks": fields[19],
                    },
                    "parent": int(fields[1]),
                    "uid": uid,
                    "cwd": None,
                }
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as error:
                if not entry.exists():
                    continue
                try:
                    comm = (entry / "comm").read_text(errors="surrogateescape").strip()
                except OSError:
                    comm = "?"
                missing.append(
                    {
                        "pid": pid,
                        "comm": comm,
                        "identity": None,
                        "items": [
                            {
                                "item": "stat",
                                "error": type(error).__name__,
                                "errno": error.errno,
                            }
                        ],
                        "related": associated,
                    }
                )
    seeds = processes.scoped_processes(root, view)
    known.update((processes.identity_key(item), item) for item in seeds)
    processes.expand_tracked(known, view)
    inspected: dict[str, dict[str, Any]] = {}
    while True:
        before = len(inspected)
        for node in list(view.values()):
            key = processes.identity_key(node["identity"])
            if key in inspected or (
                node.get("uid") not in {None, os.getuid()}
                and key not in known
                and node["identity"]["pid"] not in managed_pids
            ):
                continue
            result = _inspect(node, root, proc)
            inspected[key] = {}
            if result:
                key = processes.identity_key(result["identity"])
                inspected[key] = result
                if result["identity"] != node["identity"]:
                    view[result["identity"]["pid"]] = {
                        **node,
                        "identity": result["identity"],
                        "parent": result["parent"],
                        "uid": result["uid"],
                        "cwd": None,
                        "scope_uncertain": False,
                    }
            if result and result["references"]:
                known[key] = result["identity"]
        processes.expand_tracked(known, view)
        if len(inspected) == before:
            break  # Include foreign-UID descendants discovered from fd/maps roots.
    found, opaque = [], list(missing)
    errors = [
        f"related process reference inspection failed: {row}"
        for row in missing
        if row["related"]
    ]
    for key, result in inspected.items():
        if not result:
            continue
        if result["references"]:
            found.append(
                {
                    "identity": result["identity"],
                    "comm": result["comm"],
                    "references": result["references"],
                }
            )
        if result["unreadable"]:
            row = {
                "identity": result["identity"],
                "pid": result["identity"]["pid"],
                "comm": result["comm"],
                "items": result["unreadable"],
                "related": key in known or result["identity"]["pid"] in managed_pids,
            }
            opaque.append(row)
            if row["related"]:
                errors.append(f"related process reference inspection failed: {row}")
    return {
        "path": str(root),
        "references": found,
        "uninspectable": opaque,
        "related": list(known.values()),
        "errors": errors,
    }
