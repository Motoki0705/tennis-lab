"""No-atime snapshots, verified copies and Linux atomic publication."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StorageError(RuntimeError):
    """The operation cannot establish its safety conditions."""


def read_bytes(path: Path) -> bytes:
    """Never fall back to an ordinary read if O_NOATIME is unavailable."""
    fd = os.open(path, os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        return stream.read()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(read_bytes(path))
    if not isinstance(value, dict):
        raise StorageError(f"expected a JSON object: {path}")
    return value


def names(path: Path) -> list[str]:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOATIME | os.O_NOFOLLOW)
    try:
        with os.scandir(fd) as entries:
            return sorted(entry.name for entry in entries)
    finally:
        os.close(fd)


def sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOATIME | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def rename(source: Path, target: Path, *, exchange: bool = False) -> None:
    """RENAME_NOREPLACE or RENAME_EXCHANGE; never overwrite a successor."""
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.renameat2
    call.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(source), -100, os.fsencode(target), 2 if exchange else 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))
    sync_dir(source.parent)
    if target.parent != source.parent:
        sync_dir(target.parent)


def real_directory(path: Path, base: Path) -> None:
    """Reject symlinks in every managed path component, including its root."""
    relative = path.relative_to(base)
    current = base
    for part in ("", *relative.parts):
        current = current / part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise StorageError(
                f"real directory required (migration may be pending): {current}"
            )
        if info.st_dev != base.stat().st_dev:
            raise StorageError(f"unexpected nested filesystem: {current}")


def stamp(info: os.stat_result) -> dict[str, Any]:
    kind = (
        "file"
        if stat.S_ISREG(info.st_mode)
        else "dir"
        if stat.S_ISDIR(info.st_mode)
        else "link"
        if stat.S_ISLNK(info.st_mode)
        else "unsupported"
    )
    return {
        "kind": kind,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "atime_ns": info.st_atime_ns,
        "dev": info.st_dev,
        "ino": info.st_ino,
        "nlink": info.st_nlink,
        "blocks": info.st_blocks,
        "mode": stat.S_IMODE(info.st_mode),
    }


def identity(entry: dict[str, Any]) -> tuple[Any, ...]:
    # Directory atime/size/link count are not content. Directory mtime/ctime
    # still detect additions, removals and replacements during a snapshot.
    keys: tuple[str, ...] = ("kind", "dev", "ino", "mtime_ns", "ctime_ns", "mode")
    if entry["kind"] != "dir":
        keys += ("size", "nlink", "link")
    return tuple(entry.get(key) for key in keys)


@dataclass
class Snapshot:
    entries: dict[str, dict[str, Any]]

    @property
    def bytes(self) -> int:
        return sum(e["size"] for e in self.entries.values() if e["kind"] == "file")

    @property
    def last_access_ns(self) -> int | None:
        times = [e["atime_ns"] for e in self.entries.values() if e["kind"] == "file"]
        return max(times) if times else None

    @property
    def inodes(self) -> set[tuple[int, int]]:
        return {(e["dev"], e["ino"]) for e in self.entries.values()}

    def allocation(self) -> dict[str, int]:
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for e in self.entries.values():
            if e["kind"] == "file":
                groups.setdefault((e["dev"], e["ino"]), []).append(e)
        external = [es[0] for es in groups.values() if es[0]["nlink"] > len(es)]
        return {
            "logical_bytes": self.bytes,
            "unique_allocated_bytes": sum(
                es[0]["blocks"] * 512 for es in groups.values()
            ),
            "estimated_reclaim_bytes": sum(
                es[0]["blocks"] * 512
                for es in groups.values()
                if es[0]["nlink"] == len(es)
            ),
            "external_hardlink_inodes": len(external),
            "external_hardlink_bytes": sum(e["size"] for e in external),
        }

    def same_metadata(self, other: Snapshot, *, moved: bool = False) -> bool:
        if self.entries.keys() != other.entries.keys():
            return False
        for name, entry in self.entries.items():
            other_entry = other.entries[name]
            if moved and name == ".":
                keys = ("dev", "ino", "mode", "mtime_ns")
                if tuple(entry[k] for k in keys) != tuple(other_entry[k] for k in keys):
                    return False
            elif identity(entry) != identity(other_entry):
                return False
        return True

    def content(self) -> dict[str, tuple[Any, ...]]:
        return {
            name: (
                e["kind"],
                e.get("sha256"),
                e.get("link"),
                e["size"] if e["kind"] == "file" else None,
                e["mode"],
            )
            for name, e in self.entries.items()
        }


def snapshot(path: Path, *, hashes: bool = False) -> Snapshot:
    """Walk pinned directory FDs, rejecting mounts, special files and symlinks."""
    entries: dict[str, dict[str, Any]] = {}
    inode_hashes: dict[tuple[int, int], tuple[tuple[Any, ...], str]] = {}
    root = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOATIME | os.O_NOFOLLOW)
    device = os.fstat(root).st_dev

    def visit(fd: int, relative: str) -> None:
        before = stamp(os.fstat(fd))
        entries[relative or "."] = before
        with os.scandir(fd) as iterator:
            children = sorted(entry.name for entry in iterator)
        for name in children:
            label = f"{relative}/{name}" if relative else name
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            entry = stamp(info)
            if info.st_dev != device:
                raise StorageError(f"nested mount in unit: {path / label}")
            if entry["kind"] == "dir":
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOATIME | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
                try:
                    if (os.fstat(child).st_dev, os.fstat(child).st_ino) != (
                        info.st_dev,
                        info.st_ino,
                    ):
                        raise StorageError(
                            f"directory changed while opening: {path / label}"
                        )
                    visit(child, label)
                finally:
                    os.close(child)
                continue
            if entry["kind"] == "link":
                raise StorageError(
                    f"symlinks inside units are unsupported: readlink cannot preserve symlink atime on D: {path / label}"
                )
            elif entry["kind"] == "file" and hashes:
                key = (info.st_dev, info.st_ino)
                cached = inode_hashes.get(key)
                if cached is not None and cached[0] == identity(entry):
                    entry["sha256"] = cached[1]
                else:
                    child = os.open(
                        name, os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW, dir_fd=fd
                    )
                    with os.fdopen(child, "rb") as stream:
                        if identity(stamp(os.fstat(stream.fileno()))) != identity(
                            entry
                        ):
                            raise StorageError(
                                f"file changed while opening: {path / label}"
                            )
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                        if identity(stamp(os.fstat(stream.fileno()))) != identity(
                            entry
                        ):
                            raise StorageError(
                                f"file changed while hashing: {path / label}"
                            )
                    entry["sha256"] = digest
                    inode_hashes[key] = (identity(entry), digest)
            elif entry["kind"] != "file":
                raise StorageError(f"unsupported entry: {path / label}")
            entries[label] = entry
            after_entry = stamp(os.stat(name, dir_fd=fd, follow_symlinks=False))
            if identity(after_entry) != identity(entry):
                raise StorageError(f"entry changed during snapshot: {path / label}")
        if identity(before) != identity(stamp(os.fstat(fd))):
            raise StorageError(f"directory changed during snapshot: {path / relative}")

    try:
        visit(root, "")
    finally:
        os.close(root)
    result = Snapshot(entries)
    if hashes:
        after = snapshot(path)
        if not result.same_metadata(after):
            raise StorageError(f"unit changed during hashing: {path}")
    return result


def copy_tree(
    source: Path, target: Path, expected: Snapshot, *, now_ns: int | None = None
) -> None:
    """Copy to a new staging tree, preserving hardlinks only inside this unit."""
    target.mkdir(mode=0o700)
    links: dict[tuple[int, int], Path] = {}
    for name, entry in expected.entries.items():
        if name == ".":
            continue
        destination = target / name
        if entry["kind"] == "dir":
            destination.mkdir(mode=0o700)
        else:
            key = (entry["dev"], entry["ino"])
            if key in links:
                os.link(links[key], destination)
            else:
                fd = os.open(source / name, os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW)
                with (
                    os.fdopen(fd, "rb") as incoming,
                    destination.open("xb") as outgoing,
                ):
                    if identity(stamp(os.fstat(incoming.fileno()))) != identity(entry):
                        raise StorageError(
                            f"source changed before copying: {source / name}"
                        )
                    while block := incoming.read(8 * 1024 * 1024):
                        outgoing.write(block)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                os.chmod(destination, entry["mode"])
                os.utime(
                    destination,
                    ns=(
                        now_ns if now_ns is not None else entry["atime_ns"],
                        entry["mtime_ns"],
                    ),
                )
                links[key] = destination
    for name, entry in reversed(list(expected.entries.items())):
        if entry["kind"] == "dir":
            directory = target if name == "." else target / name
            os.chmod(directory, entry["mode"])
            os.utime(
                directory,
                ns=(
                    now_ns if now_ns is not None else entry["atime_ns"],
                    entry["mtime_ns"],
                ),
            )
            sync_dir(directory)
