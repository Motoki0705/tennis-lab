"""Full-entry deletion plans with atomic claims and retained failure evidence."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import filesystem as fs
from .leases import require_write_lease, write_lease


def remove_verified_tree(
    path: Path,
    expected: fs.Snapshot,
    *,
    retained: Path,
    retained_identity: tuple[int, int],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Claim each name before its final verification, preserving mismatches."""
    if retained.parent != path.parent or not retained.name.startswith(
        ".storage-tier-retained-"
    ):
        raise fs.StorageError("invalid retained-entry directory")
    plan = {name: dict(entry) for name, entry in expected.entries.items()}
    root_identity = (plan["."]["dev"], plan["."]["ino"])
    children: dict[str, set[str]] = {
        n: set() for n, e in plan.items() if e["kind"] == "dir"
    }
    for name in plan:
        if name != ".":
            children[str(Path(name).parent)].add(Path(name).name)
    inode_state: dict[tuple[int, int], dict[str, int]] = {}
    for entry in plan.values():
        if entry["kind"] == "file":
            key = (entry["dev"], entry["ino"])
            state = {"ctime_ns": entry["ctime_ns"], "nlink": entry["nlink"]}
            if key in inode_state and inode_state[key] != state:
                raise fs.StorageError("inconsistent hardlink metadata in removal plan")
            inode_state[key] = state
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NOATIME
    parent = root = vault = -1
    root_token: str | None = None

    def named(fd: int, name: str, identity: tuple[int, int]) -> bool:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        return stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == identity

    def check_locations() -> None:
        if not named(parent, retained.name, retained_identity):
            raise fs.StorageError("retained directory identity changed")
        if not (
            named(parent, path.name, root_identity)
            if root_token is None
            else named(vault, root_token, root_identity)
        ):
            raise fs.StorageError("quarantine identity changed during removal")

    def matching(
        entry: dict[str, Any], observed: dict[str, Any], *, renamed: bool = False
    ) -> bool:
        adjusted = {**entry, "ctime_ns": observed["ctime_ns"]} if renamed else entry
        return fs.identity(adjusted) == fs.identity(observed) and (
            entry["kind"] != "file" or entry["atime_ns"] == observed["atime_ns"]
        )

    def current_entry(label: str) -> dict[str, Any]:
        entry = plan[label]
        if entry["kind"] == "file":
            return {**entry, **inode_state[(entry["dev"], entry["ino"])]}
        return entry

    def report(event: str, label: str, **details: Any) -> None:
        if progress is not None:
            progress(
                {
                    "event": event,
                    "entry": label,
                    "expected": dict(current_entry(label)),
                    **details,
                }
            )

    def check_entry(fd: int, name: str, label: str) -> None:
        if not matching(
            current_entry(label),
            fs.stamp(os.stat(name, dir_fd=fd, follow_symlinks=False)),
        ):
            raise fs.StorageError(f"verified entry changed before removal: {label}")

    def check_directory(
        fd: int, label: str, remaining: set[str], *, inventory: bool = False
    ) -> None:
        check_locations()
        if not matching(plan[label], fs.stamp(os.fstat(fd))):
            raise fs.StorageError(f"verified directory changed during removal: {label}")
        if inventory:
            with os.scandir(fd) as entries:
                actual = {entry.name for entry in entries}
            if actual != remaining:
                raise fs.StorageError(
                    f"verified entry set changed during removal: {label}"
                )

    def parent_after_claim(fd: int, label: str) -> None:
        observed = fs.stamp(os.fstat(fd))
        if any(observed[k] != plan[label][k] for k in ("kind", "dev", "ino", "mode")):
            raise fs.StorageError(
                f"directory identity/mode changed during removal: {label}"
            )
        plan[label] = observed

    @contextmanager
    def claimed(
        fd: int, name: str, label: str, parent_label: str | None
    ) -> Iterator[str]:
        nonlocal root_token
        token = uuid.uuid4().hex
        report("claim_intent", label, retained_name=token)
        fs.rename_at(fd, name, vault, token)
        if label == ".":
            root_token = token
        try:
            if parent_label is not None:
                parent_after_claim(fd, parent_label)
            report(
                "claimed",
                label,
                retained_name=token,
                actual=fs.stamp(os.stat(token, dir_fd=vault, follow_symlinks=False)),
            )
            yield token
        except BaseException:
            try:
                os.stat(token, dir_fd=vault, follow_symlinks=False)
            except FileNotFoundError:
                pass  # Unlink may have completed before completion journaling failed.
            else:
                try:
                    fs.rename_at(vault, token, fd, name)
                except OSError as exc:
                    report(
                        "retained", label, retained_name=token, restore_error=str(exc)
                    )
                else:
                    if label == ".":
                        root_token = None
                    report("restored", label, retained_name=token)
            raise

    def accept_claim(token: str, label: str) -> dict[str, Any]:
        observed = fs.stamp(os.stat(token, dir_fd=vault, follow_symlinks=False))
        entry = current_entry(label)
        if not matching(entry, observed, renamed=True):
            raise fs.StorageError(f"claimed entry is not the verified entry: {label}")
        # Rename changes ctime, including other hardlink names for the inode.
        if entry["kind"] == "file":
            inode_state[(entry["dev"], entry["ino"])]["ctime_ns"] = observed["ctime_ns"]
            return current_entry(label)
        else:
            entry["ctime_ns"] = observed["ctime_ns"]
        return entry

    def clear(fd: int, label: str, mount_id: int) -> None:
        remaining = set(children[label])
        check_directory(fd, label, remaining, inventory=True)
        for name in sorted(remaining):
            check_directory(fd, label, remaining)
            child_label = name if label == "." else f"{label}/{name}"
            check_entry(fd, name, child_label)
            entry = plan[child_label]
            if entry["kind"] == "dir":
                child = os.open(name, flags, dir_fd=fd)
                try:
                    if (
                        not matching(entry, fs.stamp(os.fstat(child)))
                        or fs.fd_mount_id(child) != mount_id
                    ):
                        raise fs.StorageError(
                            "directory identity/mount changed during removal"
                        )
                    clear(child, child_label, mount_id)
                    with claimed(fd, name, child_label, label) as token:
                        remaining.remove(name)
                        accept_claim(token, child_label)
                        check_directory(child, child_label, set(), inventory=True)
                        check_directory(fd, label, remaining)
                        report("remove_intent", child_label, retained_name=token)
                        check_entry(vault, token, child_label)
                        os.rmdir(token, dir_fd=vault)
                        os.fsync(vault)
                        report("removed", child_label, retained_name=token)
                finally:
                    os.close(child)
            elif entry["kind"] == "file":
                with claimed(fd, name, child_label, label) as token:
                    remaining.remove(name)
                    entry = accept_claim(token, child_label)
                    child = os.open(
                        token, os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW, dir_fd=vault
                    )
                    with os.fdopen(child, "rb") as stream:
                        phase, unlinked = "lease_acquire", False
                        try:
                            with write_lease(child):
                                phase = "leased_verification"
                                if fs.fd_mount_id(child) != mount_id or not matching(
                                    entry, fs.stamp(os.fstat(child))
                                ):
                                    raise fs.StorageError(
                                        f"claimed file identity/mount changed: {child_label}"
                                    )
                                report(
                                    "remove_intent", child_label, retained_name=token
                                )
                                digest = hashlib.file_digest(
                                    stream, "sha256"
                                ).hexdigest()
                                if digest != entry.get("sha256") or not matching(
                                    entry, fs.stamp(os.fstat(child))
                                ):
                                    raise fs.StorageError(
                                        f"verified file changed before unlink: {child_label}"
                                    )
                                check_directory(fd, label, remaining)
                                check_entry(vault, token, child_label)
                                # Bind the leased descriptor to the last pathname
                                # check too: replacing that name can unlink the
                                # original inode and invalidate its metadata.
                                if not matching(entry, fs.stamp(os.fstat(child))):
                                    raise fs.StorageError(
                                        "leased inode changed before unlink"
                                    )
                                phase = "before_unlink"
                                require_write_lease(child)
                                os.unlink(token, dir_fd=vault)
                                unlinked = True
                                phase = "after_unlink"
                                try:
                                    require_write_lease(child)
                                finally:
                                    os.fsync(vault)
                                    report("removed", child_label, retained_name=token)
                                after = fs.stamp(os.fstat(child))
                                if after["nlink"] != entry["nlink"] - 1 or any(
                                    after[key] != entry[key]
                                    for key in (
                                        "kind",
                                        "dev",
                                        "ino",
                                        "mode",
                                        "size",
                                        "mtime_ns",
                                        "atime_ns",
                                    )
                                ):
                                    raise fs.StorageError(
                                        f"leased inode/link count changed during removal: {child_label}"
                                    )
                                inode_state[(entry["dev"], entry["ino"])] = {
                                    "nlink": after["nlink"],
                                    "ctime_ns": after["ctime_ns"],
                                }
                        except (OSError, fs.StorageError) as exc:
                            report(
                                "file_removal_failed",
                                child_label,
                                retained_name=token,
                                phase=phase,
                                unlinked=unlinked,
                                error=str(exc),
                            )
                            raise
            else:
                raise fs.StorageError(
                    f"unverified entry type during removal: {child_label}"
                )
            check_directory(fd, label, remaining)
        check_directory(fd, label, set(), inventory=True)

    try:
        parent = os.open(path.parent, flags)
        root = os.open(path.name, flags, dir_fd=parent)
        vault = os.open(retained.name, flags, dir_fd=parent)
        check_locations()
        root_info = os.fstat(root)
        vault_info = os.fstat(vault)
        if (root_info.st_dev, root_info.st_ino) != root_identity or (
            vault_info.st_dev,
            vault_info.st_ino,
        ) != retained_identity:
            raise fs.StorageError("quarantine/retained identity changed before removal")
        mount_id = fs.fd_mount_id(root)
        if expected.mount_id != mount_id or any(
            fs.fd_mount_id(fd) != mount_id for fd in (parent, vault)
        ):
            raise fs.StorageError("quarantine/retained mount changed before removal")
        with os.scandir(vault) as entries:
            if next(entries, None) is not None:
                raise fs.StorageError("retained directory must initially be empty")
        observed = fs.snapshot(path)
        if expected.entries.keys() != observed.entries.keys() or any(
            not matching(e, observed.entries[n]) for n, e in expected.entries.items()
        ):
            raise fs.StorageError("verified tree changed before removal")
        clear(root, ".", mount_id)
        with claimed(parent, path.name, ".", None) as token:
            accept_claim(token, ".")
            check_directory(root, ".", set(), inventory=True)
            report("remove_intent", ".", retained_name=token)
            check_entry(vault, token, ".")
            os.rmdir(token, dir_fd=vault)
            os.fsync(vault)
            report("removed", ".", retained_name=token)
        if os.path.lexists(path):
            raise fs.StorageError("a successor quarantine appeared during removal")
        if not named(parent, retained.name, retained_identity):
            raise fs.StorageError("retained directory identity changed before cleanup")
        os.rmdir(retained.name, dir_fd=parent)  # Only empty tool-owned containers.
        os.fsync(parent)
    finally:
        for fd in (vault, root, parent):
            if fd >= 0:
                os.close(fd)
