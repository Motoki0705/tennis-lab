"""Fail-closed observations of the execution user's descriptors and mappings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .filesystem import Snapshot, StorageError, names, read_bytes


@dataclass
class Usage:
    busy: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    uid: int = field(default_factory=os.geteuid)
    inspected_processes: int = 0
    excluded_uids: dict[str, int] = field(default_factory=dict)

    def scope(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "inspected_processes": self.inspected_processes,
            "excluded_other_uid": sum(self.excluded_uids.values()),
            "excluded_uids": self.excluded_uids,
        }


class Inspector:
    def __init__(self, proc: Path = Path("/proc")) -> None:
        self.proc = proc

    def effective_uid(self, process: Path) -> int:
        status = process / "status"
        if self.proc == Path("/proc"):
            # Kernel-generated process metadata is not dataset access history.
            # Normal reads can identify a nondumpable same-UID process even
            # when /proc ownership was reset to root and O_NOATIME is denied.
            fd = os.open(status, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                value = stream.read()
        else:
            value = read_bytes(status)
        for line in value.decode(errors="surrogateescape").splitlines():
            if line.startswith("Uid:"):
                uids = line.split()[1:]
                if len(uids) != 4 or any(not value.isdecimal() for value in uids):
                    break
                return int(uids[1])
        raise StorageError(f"missing/malformed process UID: {process}")

    def inspect(self, paths: tuple[Path, ...], tree: Snapshot) -> Usage:
        result = Usage()
        inodes = tree.inodes
        # /proc itself belongs to root and rejects O_NOATIME for an ordinary
        # user. Its synthetic kernel directory has no dataset access history.
        # Real directories, including injected test inventories, use NOATIME.
        processes = (
            sorted(p.name for p in self.proc.iterdir())
            if self.proc == Path("/proc")
            else names(self.proc)
        )
        for process_name in processes:
            process = self.proc / process_name
            if not process.name.isdecimal():
                continue
            try:
                uid = self.effective_uid(process)
                if uid != result.uid:
                    uid_key = str(uid)
                    result.excluded_uids[uid_key] = (
                        result.excluded_uids.get(uid_key, 0) + 1
                    )
                    continue
                result.inspected_processes += 1
                for descriptor_name in names(process / "fd"):
                    descriptor = process / "fd" / descriptor_name
                    try:
                        info = descriptor.stat()
                        if (info.st_dev, info.st_ino) in inodes:
                            result.busy.append(
                                f"open-fd: pid={process.name} fd={descriptor.name}"
                            )
                    except FileNotFoundError:
                        continue
                for name in ("cwd", "root"):
                    target = Path(
                        os.readlink(process / name).removesuffix(" (deleted)")
                    )
                    if any(target.is_relative_to(path) for path in paths):
                        result.busy.append(f"process-{name}: pid={process.name}")
                # A process may close its fd while retaining an mmap, including
                # a mapping reached through a hardlink outside this unit.
                for line in (
                    read_bytes(process / "maps")
                    .decode(errors="surrogateescape")
                    .splitlines()
                ):
                    columns = line.split(maxsplit=5)
                    if len(columns) < 5:
                        raise StorageError(f"malformed maps: {process}")
                    major, minor = columns[3].split(":")
                    key = (os.makedev(int(major, 16), int(minor, 16)), int(columns[4]))
                    if key in inodes:
                        result.busy.append(f"open-mmap: pid={process.name}")
            except (FileNotFoundError, ProcessLookupError) as exc:
                if process.exists():
                    result.unknown.append(
                        f"pid={process.name}: process metadata unavailable: {exc}"
                    )
                continue
            except (OSError, ValueError, StorageError) as exc:
                if process.exists():
                    result.unknown.append(f"pid={process.name}: {exc}")
        return result
