"""Fail-closed observations of the execution user's descriptors and mappings."""

from __future__ import annotations

import errno
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
    excluded_nondumpable: list[dict[str, Any]] = field(default_factory=list)
    excluded_capabilities: list[dict[str, Any]] = field(default_factory=list)
    exited_processes: int = 0

    def scope(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "inspected_processes": self.inspected_processes,
            "excluded_other_uid": sum(self.excluded_uids.values()),
            "excluded_uids": self.excluded_uids,
            "excluded_nondumpable_count": len(self.excluded_nondumpable),
            "excluded_nondumpable": self.excluded_nondumpable,
            "excluded_capabilities_count": len(self.excluded_capabilities),
            "excluded_capabilities": self.excluded_capabilities,
            "exited_processes": self.exited_processes,
        }


class Inspector:
    def __init__(self, proc: Path = Path("/proc"), *, uid: int | None = None) -> None:
        self.proc = proc
        self.uid = os.geteuid() if uid is None else uid

    def metadata(self, process: Path, name: str) -> bytes:
        path = process / name
        if self.proc == Path("/proc"):
            # Kernel-generated process metadata is not dataset access history.
            # Normal reads can identify a nondumpable same-UID process even
            # when /proc ownership was reset to root and O_NOATIME is denied.
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                return stream.read()
        return read_bytes(path)

    def status(self, process: Path) -> dict[str, str]:
        return {
            key: value.strip()
            for line in self.metadata(process, "status")
            .decode(errors="surrogateescape")
            .splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
        }

    @staticmethod
    def uids(process: Path, status: dict[str, str]) -> tuple[int, ...]:
        values = status.get("Uid", "").split()
        if len(values) == 4 and all(value.isdecimal() for value in values):
            return tuple(int(value) for value in values)
        raise StorageError(f"missing/malformed process UID: {process}")

    def process_names(self) -> list[str]:
        return (
            sorted(p.name for p in self.proc.iterdir())
            if self.proc == Path("/proc")
            else names(self.proc)
        )

    def entries(self, process: Path, name: str) -> list[str]:
        if self.proc == Path("/proc"):
            with os.scandir(process / name) as entries:
                return sorted(entry.name for entry in entries)
        return names(process / name)

    def descriptors(self, process: Path) -> list[str]:
        return self.entries(process, "fd")

    def exited(self, process: Path, status: dict[str, str]) -> bool:
        # A zombie thread-group leader can still have live sibling threads.
        # Confirm the task inventory and recheck identity/state after the read.
        def closed(value: dict[str, str]) -> bool:
            state = value.get("State", "").split()
            return bool(
                state and state[0] in {"Z", "X", "x"} and value.get("Threads") == "1"
            )

        if not closed(status):
            return False
        generation = self.starttime(process)
        before = self.status(process)
        if not closed(before) or self.entries(process, "task") != [process.name]:
            return False
        after = self.status(process)
        return (
            closed(after)
            and self.uids(process, status)
            == self.uids(process, before)
            == self.uids(process, after)
            and self.starttime(process) == generation
        )

    def starttime(self, process: Path) -> int:
        value = self.metadata(process, "stat").decode(errors="surrogateescape")
        try:
            prefix, fields = value.rsplit(")", 1)
            if int(prefix.split("(", 1)[0].strip()) != int(process.name):
                raise ValueError("PID differs")
            result = int(
                fields.split()[19]
            )  # field 22; comm may contain spaces/parentheses
            if result < 0:
                raise ValueError("negative starttime")
            return result
        except (ValueError, IndexError) as exc:
            raise StorageError(f"invalid process identity: {process}") from exc

    def nondumpable_evidence(self, process: Path) -> dict[str, Any]:
        # proc_pid(5): non-user-dumpable processes' entries become root:root.
        # Permission errors alone also arise for dumpable/capability-bearing
        # processes, so require independent ownership plus all three denials.
        before = self.status(process)
        if self.uid == 0 or self.uids(process, before) != (self.uid,) * 4:
            raise StorageError("cannot establish nondumpability from UID metadata")
        if before.get("Dumpable") not in {None, "0", "2"}:
            raise StorageError("process reports dumpable or invalid Dumpable value")
        start = self.starttime(process)

        def owners() -> None:
            for name in ("fd", "maps", "cwd"):
                info = (process / name).lstat()
                if (info.st_uid, info.st_gid) != (0, 0):
                    raise StorageError(
                        f"{name} is not root-owned; nondumpability is unproven"
                    )

        owners()
        denied = {}
        for name in ("fd", "maps", "cwd"):
            try:
                if name == "fd":
                    self.descriptors(process)
                elif name == "maps":
                    self.metadata(process, name)
                else:
                    os.readlink(process / name)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EPERM}:
                    raise
                denied[name] = errno.errorcode[exc.errno]
            else:
                raise StorageError(f"{name} is readable; nondumpability is unproven")
        after = self.status(process)
        owners()
        if (
            self.uids(process, after) != (self.uid,) * 4
            or after.get("Dumpable") != before.get("Dumpable")
            or after.get("Name") != before.get("Name")
            or self.starttime(process) != start
        ):
            raise StorageError(
                "process identity or credentials changed during exclusion"
            )
        return {
            "kind": "nondumpable",
            "pid": int(process.name),
            "comm": after.get("Name"),
            "starttime_ticks": start,
            "proc_owner_uid": 0,
            "proc_owner_gid": 0,
            "denied": denied,
        }

    @staticmethod
    def permitted(process: Path, status: dict[str, str]) -> int:
        value = status.get("CapPrm", "")
        if len(value) != 16 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise StorageError(f"missing/malformed process CapPrm: {process}")
        return int(value, 16)

    @staticmethod
    def user_namespace(process: Path) -> tuple[int, int]:
        # Follow the namespace handle, not the per-PID proc symlink inode.
        # Equal uid_map/gid_map contents do not prove namespace identity.
        try:
            info = (process / "ns/user").stat()
        except PermissionError as exc:
            raise StorageError(
                f"user namespace identity cannot be read: {process}: {exc}"
            ) from exc
        return info.st_dev, info.st_ino

    def capability_evidence(self, process: Path) -> dict[str, Any]:
        caller = self.proc / "self"
        before, caller_before = self.status(process), self.status(caller)
        if (
            self.uids(process, before) != (self.uid,) * 4
            or self.uids(caller, caller_before) != (self.uid,) * 4
        ):
            raise StorageError("cannot establish same-UID capability exclusion")
        target_caps = self.permitted(process, before)
        caller_caps = self.permitted(caller, caller_before)
        masks = f"target CapPrm={target_caps:016x}, inspector CapPrm={caller_caps:016x}"
        if target_caps & ~caller_caps == 0:
            raise StorageError(
                f"target CapPrm is a subset of inspector CapPrm: {masks}"
            )
        start = self.starttime(process)
        try:
            namespace = self.user_namespace(process)
            if namespace != self.user_namespace(caller):
                raise StorageError(
                    "capability comparison requires the same user namespace"
                )
        except (OSError, StorageError) as exc:
            raise StorageError(f"{exc}; {masks}") from exc
        denied = {}
        for name in ("maps", "cwd"):
            try:
                if name == "maps":
                    self.metadata(process, name)
                else:
                    os.readlink(process / name)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EPERM}:
                    raise
                denied[name] = errno.errorcode[exc.errno]
            else:
                raise StorageError(f"{name} is readable; capability denial is unproven")
        after, caller_after = self.status(process), self.status(caller)
        if (
            self.uids(process, after) != self.uids(process, before)
            or self.uids(caller, caller_after) != self.uids(caller, caller_before)
            or self.permitted(process, after) != target_caps
            or self.permitted(caller, caller_after) != caller_caps
            or after.get("Name") != before.get("Name")
            or self.starttime(process) != start
            or self.user_namespace(process) != namespace
            or self.user_namespace(caller) != namespace
        ):
            raise StorageError("process identity, capabilities or namespace changed")
        return {
            "kind": "permitted-capabilities",
            "pid": int(process.name),
            "comm": after.get("Name"),
            "starttime_ticks": start,
            "target_cap_prm": f"{target_caps:016x}",
            "inspector_cap_prm": f"{caller_caps:016x}",
            "user_namespace": {"device": namespace[0], "inode": namespace[1]},
            "denied": denied,
        }

    def exclusion_evidence(self, process: Path) -> dict[str, Any]:
        try:
            return self.nondumpable_evidence(process)
        except StorageError as nondumpable_error:
            try:
                return self.capability_evidence(process)
            except (OSError, ValueError, StorageError) as capability_error:
                raise StorageError(
                    f"{nondumpable_error}; {capability_error}"
                ) from capability_error

    def inspect(self, paths: tuple[Path, ...], tree: Snapshot) -> Usage:
        result = Usage(uid=self.uid)
        inodes = tree.inodes
        # /proc itself belongs to root and rejects O_NOATIME for an ordinary
        # user. Its synthetic kernel directory has no dataset access history.
        # Real directories, including injected test inventories, use NOATIME.
        processes = self.process_names()
        for process_name in processes:
            process = self.proc / process_name
            if not process.name.isdecimal():
                continue
            busy_start = len(result.busy)
            inspecting = False
            try:
                status = self.status(process)
                uid = self.uids(process, status)[1]
                if uid != result.uid:
                    uid_key = str(uid)
                    result.excluded_uids[uid_key] = (
                        result.excluded_uids.get(uid_key, 0) + 1
                    )
                    continue
                if self.exited(process, status):
                    result.exited_processes += 1
                    continue
                result.inspected_processes += 1
                inspecting = True
                for descriptor_name in self.descriptors(process):
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
                    self.metadata(process, "maps")
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
                else:
                    result.exited_processes += 1
                    result.inspected_processes -= int(inspecting)
                    del result.busy[busy_start:]
                continue
            except (OSError, ValueError, StorageError) as exc:
                if process.exists():
                    if isinstance(exc, OSError) and exc.errno in {
                        errno.EACCES,
                        errno.EPERM,
                    }:
                        try:
                            evidence = self.exclusion_evidence(process)
                        except (OSError, ValueError, StorageError) as proof_error:
                            result.unknown.append(
                                f"pid={process.name}: {exc}; {proof_error}"
                            )
                            continue
                        excluded = (
                            result.excluded_nondumpable
                            if evidence["kind"] == "nondumpable"
                            else result.excluded_capabilities
                        )
                        excluded.append(evidence)
                        result.inspected_processes -= int(inspecting)
                        del result.busy[busy_start:]
                        continue
                    result.unknown.append(f"pid={process.name}: {exc}")
        return result
