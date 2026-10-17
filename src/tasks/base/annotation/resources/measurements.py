"""Allocated storage and Linux PSS. No RSS-as-PSS or symlink traversal fallback."""

from __future__ import annotations

import json
import os
import stat as stat_modes
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..workers.processes import ProcessIdentity, process_snapshot
from .contracts import GPUUsage, ProcessUsage, StorageUsage


def gpu_usage(pids: set[int]) -> GPUUsage:
    """Read-only telemetry. An unsupported backend is recorded, never treated as 0."""
    try:
        devices = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        device_values = [
            tuple(int(part.strip()) for part in line.split(","))
            for line in devices.stdout.splitlines()
            if line.strip()
        ]
        process_values = [
            tuple(int(part.strip()) for part in line.split(","))
            for line in applications.stdout.splitlines()
            if line.strip()
        ]
        if not device_values or any(
            len(row) != 2 for row in device_values + process_values
        ):
            raise ValueError("unexpected nvidia-smi CSV shape")
        return GPUUsage(
            status="available",
            owned_vram_bytes=sum(
                amount * 1024**2 for pid, amount in process_values if pid in pids
            ),
            device_vram_bytes=sum(row[0] * 1024**2 for row in device_values),
            maximum_utilization_percent=float(max(row[1] for row in device_values)),
            reason="owned bytes cover only compute PIDs reported by nvidia-smi; device bytes include other processes",
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return GPUUsage(
            status="unavailable",
            owned_vram_bytes=None,
            device_vram_bytes=None,
            maximum_utilization_percent=None,
            reason=f"{type(error).__name__}: {error}",
        )


def _parallel_metadata_roots(roots: tuple[str, ...]) -> tuple[str, ...]:
    """Use bounded I/O concurrency for 9p mounts, including WSL host volumes."""
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, separator, filesystem = line.partition(" - ")
        if not separator:
            raise ValueError("invalid Linux mount information")
        mount = fields.split()[4]
        for escaped, decoded in (
            (r"\040", " "),
            (r"\011", "\t"),
            (r"\012", "\n"),
            (r"\134", "\\"),
        ):
            mount = mount.replace(escaped, decoded)
        mounts.append((Path(mount), filesystem.split()[0]))
    mounts.sort(key=lambda item: len(item[0].parts), reverse=True)
    return tuple(
        root
        for root in roots
        if next(kind for mount, kind in mounts if Path(root).is_relative_to(mount))
        == "9p"
    )


def _metadata_entry(
    path: str,
) -> tuple[str, os.stat_result, tuple[str, ...]] | None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    children: tuple[str, ...] = ()
    if stat_modes.S_ISDIR(metadata.st_mode):
        try:
            with os.scandir(path) as entries:
                children = tuple(item.path for item in entries)
        except FileNotFoundError:
            pass
    return path, metadata, children


def _parallel_allocations(
    roots: tuple[str, ...],
) -> Iterator[tuple[str, os.stat_result]]:
    # Bound both simultaneous host requests and queued futures. No metadata is
    # reused between inventories, and permission/I/O errors remain fatal.
    pending = list(roots)
    with ThreadPoolExecutor(max_workers=8) as executor:
        while pending:
            batch = [pending.pop() for _ in range(min(64, len(pending)))]
            for result in executor.map(_metadata_entry, batch):
                if result is not None:
                    path, metadata, children = result
                    yield path, metadata
                    pending.extend(children)


def allocations(roots: tuple[str, ...]) -> Iterator[tuple[str, os.stat_result]]:
    parallel = _parallel_metadata_roots(roots)
    for root in roots:
        if root in parallel:
            continue
        pending = [Path(root)]
        while pending:
            path = pending.pop()
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            yield str(path), stat
            if stat_modes.S_ISDIR(stat.st_mode):
                try:
                    with os.scandir(path) as entries:
                        pending.extend(Path(item.path) for item in entries)
                except FileNotFoundError:
                    continue
    yield from _parallel_allocations(parallel)


def write_baseline(path: Path, roots: tuple[str, ...]) -> None:
    """Streaming inventory, including directories and symlink metadata, never targets."""
    seen: set[tuple[int, int]] = set()
    with path.open("xb") as handle:
        for name, stat in allocations(roots):
            key = stat.st_dev, stat.st_ino
            if key in seen or name == str(path):
                continue
            seen.add(key)
            handle.write(
                (json.dumps([*key, stat.st_blocks * 512, name]) + "\n").encode()
            )
        handle.flush()
        os.fsync(handle.fileno())


def free_bytes(roots: tuple[str, ...], *, mounts: tuple[str, ...] = ()) -> int:
    available = []
    for root in roots:
        path = Path(root)
        while not path.exists():
            path = path.parent
        stat = os.statvfs(path)
        available.append(stat.f_bavail * stat.f_frsize)
    for mount in mounts:
        path = Path(mount)
        # A detached host volume can leave an ordinary directory behind. Never
        # measure that directory's parent filesystem as the configured volume.
        if path.resolve() != path or not path.is_mount():
            raise ValueError(f"required free-space mount is unavailable: {mount}")
        stat = os.statvfs(path)
        available.append(stat.f_bavail * stat.f_frsize)
    return min(available)


def storage_usage(
    roots: tuple[str, ...],
    baseline: Path,
    prepaid: int,
    *,
    free_space_mounts: tuple[str, ...] = (),
) -> StorageUsage:
    # Keep the original anchor: deleting old files cannot credit new files elsewhere.
    prior: dict[tuple[int, int], tuple[int, str]] = {}
    with baseline.open() as handle:
        for line in handle:
            device, inode, allocated, name = json.loads(line)
            prior[(device, inode)] = (allocated, name)
    seen: set[tuple[int, int]] = set()
    total = logical = deduction = count = 0
    for name, stat in allocations(roots):
        count += 1
        logical += stat.st_size
        key = stat.st_dev, stat.st_ino
        original = prior.get(key)
        # Only the original path can authorize a deduction. A surviving hardlink
        # after anchor deletion cannot credit new files. This needs one tree scan,
        # instead of an extra lstat for every historical anchor before that scan.
        if original is not None and original[1] == name:
            deduction += min(stat.st_blocks * 512, original[0])
        if key in seen:
            continue
        seen.add(key)
        amount = stat.st_blocks * 512
        total += amount
    return StorageUsage(
        allocated_bytes=total,
        additional_bytes=total - deduction + prepaid,
        logical_bytes=logical,
        entries=count,
        unique_inodes=len(seen),
        free_bytes=free_bytes(roots, mounts=free_space_mounts),
    )


def process_usage(
    owners: tuple[ProcessIdentity, ...], observers: tuple[ProcessIdentity, ...]
) -> tuple[ProcessUsage, ...]:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    owners = tuple(owner for owner in owners if owner.boot_id == boot)
    observed = {item.pid: item for item in observers if item.boot_id == boot}
    for owner in owners:
        current = process_snapshot(owner.pid)
        if current is not None and current[0] != owner:
            raise RuntimeError("resource owner PID was recycled")
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        first = process_snapshot(int(entry.name))
        if first is None or first[1] in {"Z", "X"}:
            continue
        item = first[0]
        if observed.get(item.pid) != item and not any(
            item.group == owner.group
            and item.session == owner.session
            and item.start_ticks >= owner.start_ticks
            for owner in owners
        ):
            continue
        try:
            memory = dict(
                line.split(":", 1)
                for line in (entry / "smaps_rollup").read_text().splitlines()[1:]
            )
            io = dict(
                line.split(":", 1) for line in (entry / "io").read_text().splitlines()
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # A process can exit or change identity between stat and procfs reads.
            # Skip only a confirmed stale observation, never an unreadable live one.
            current = process_snapshot(item.pid)
            if current is None or current[0] != item or current[1] in {"Z", "X"}:
                continue
            raise
        second = process_snapshot(item.pid)
        if second is None or second[0] != item or second[1] in {"Z", "X"}:
            continue
        # Missing Pss / denied access is a measurement error, never an assumed zero.
        result.append(
            ProcessUsage(
                pid=item.pid,
                start_ticks=item.start_ticks,
                group=item.group,
                session=item.session,
                pss_bytes=int(memory["Pss"].split()[0]) * 1024,
                rss_bytes=int(memory["Rss"].split()[0]) * 1024,
                read_bytes=int(io["read_bytes"]),
                write_bytes=int(io["write_bytes"]),
            )
        )
    return tuple(sorted(result, key=lambda item: item.pid))
