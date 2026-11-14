"""Map a pinned unit through the caller's mount table before comparing aliases."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .filesystem import Snapshot, StorageError, fd_mount_id


def mount_path(raw: str) -> Path:
    if re.search(r"\\(?!040|011|012|134)", raw):
        raise StorageError("malformed mountinfo path escape")
    value = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), raw)
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or "//" in value
        or value.endswith(" (deleted)")
    ):
        raise StorageError(f"unverifiable mountinfo path: {raw}")
    return path


@dataclass(frozen=True)
class Mount:
    id: int
    device: int
    root: str
    point: str


def mounts(data: bytes) -> list[Mount]:
    result = []
    for line in data.decode(errors="surrogateescape").splitlines():
        columns = line.split()
        try:
            separator = columns.index("-")
            if separator < 6 or len(columns) < separator + 4:
                raise ValueError("missing mountinfo fields")
            mount_id, parent = int(columns[0]), int(columns[1])
            major, minor = (int(value) for value in columns[2].split(":"))
            if min(mount_id, parent, major, minor) < 0:
                raise ValueError("negative mount identity")
            device = os.makedev(major, minor)
        except (ValueError, OverflowError) as exc:
            raise StorageError("malformed mountinfo record") from exc
        result.append(Mount(mount_id, device, columns[3], columns[4]))
    return result


def filesystem_unit(
    paths: tuple[Path, ...],
    tree: Snapshot,
    metadata: Callable[[Path, str], bytes],
) -> tuple[int, Path]:
    """Use our FD's mount ID and our namespace, never a consumer's mapping."""
    root: dict[str, Any] = tree.entries["."]
    for path in reversed(paths):
        try:
            fd = os.open(path, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            continue
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != (root["dev"], root["ino"]):
                continue
            canonical = mount_path(os.readlink(f"/proc/self/fd/{fd}"))
            mount_id = fd_mount_id(fd)
            own = [
                m
                for m in mounts(metadata(Path("/proc/self"), "mountinfo"))
                if m.id == mount_id
            ]
            if len(own) != 1 or own[0].device != info.st_dev:
                raise StorageError(
                    "unit mount is missing/ambiguous in caller namespace"
                )
            record = own[0]
            try:
                relative = canonical.relative_to(mount_path(record.point))
            except ValueError as exc:
                raise StorageError("unit is outside its caller mountpoint") from exc
            return info.st_dev, mount_path(record.root) / relative
        finally:
            os.close(fd)
    raise StorageError("cannot locate snapshot root in the caller namespace")
