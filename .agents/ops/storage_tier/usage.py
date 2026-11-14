"""Fail-closed observations of queue records, descriptors and mapped inodes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .filesystem import Snapshot, StorageError, names, read_bytes


@dataclass
class Usage:
    busy: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


class Inspector:
    def __init__(self, queue: Path, proc: Path = Path("/proc")) -> None:
        self.queue = queue
        self.proc = proc

    def inspect(self, paths: tuple[Path, ...], tree: Snapshot) -> Usage:
        result = Usage()
        # Commands can acquire paths from arbitrary Python/Hydra/shell code.
        # An active or about-to-start queue record therefore blocks all units;
        # substring matching the shell command would give false assurances.
        for state in ("jobs", "running"):
            directory = self.queue / state
            try:
                for name in names(directory):
                    if name.endswith(".job"):
                        result.busy.append(f"queue-{state}: {name}")
            except (OSError, StorageError) as exc:
                result.unknown.append(f"queue inspection failed: {exc}")
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
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError, StorageError) as exc:
                if process.exists():
                    result.unknown.append(f"pid={process.name}: {exc}")
        return result
