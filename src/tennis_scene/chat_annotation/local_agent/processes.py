"""Linux worker ownership and race-safe signalling of isolated process groups."""

from __future__ import annotations

import ctypes
import os
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_LIBC = ctypes.CDLL(None, use_errno=True)
_PIDFD_OPEN = _LIBC.pidfd_open
_PIDFD_OPEN.argtypes = (ctypes.c_int, ctypes.c_uint)
_PIDFD_OPEN.restype = ctypes.c_int
_PIDFD_SIGNAL = _LIBC.pidfd_send_signal
_PIDFD_SIGNAL.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)
_PIDFD_SIGNAL.restype = ctypes.c_int


def _checked(result: int) -> int:
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _pidfd_open(pid: int) -> int:
    # Some standalone CPython builds omit os.pidfd_open even when Linux/glibc support it.
    # Use the same fd API directly, never a PID-only signalling fallback.
    return _checked(int(_PIDFD_OPEN(pid, 0)))


def _pidfd_signal(descriptor: int, number: int) -> None:
    _checked(int(_PIDFD_SIGNAL(descriptor, number, None, 0)))


def require_process_backend() -> None:
    """Fail before launching a worker when race-safe signalling is unavailable."""
    descriptor = _pidfd_open(os.getpid())
    try:
        _pidfd_signal(descriptor, 0)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    group: int
    session: int
    boot_id: str

    def document(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> ProcessIdentity:
        if set(value) != {'pid', 'start_ticks', 'group', 'session', 'boot_id'}:
            raise ValueError('Invalid worker process identity fields')
        if any(type(value[key]) is not int or value[key] <= 0 for key in ('pid', 'start_ticks', 'group', 'session')) \
                or not isinstance(value['boot_id'], str) or not value['boot_id']:
            raise ValueError('Invalid worker process identity values')
        result = cls(**value)
        if result.pid != result.group or result.pid != result.session:
            raise ValueError('Worker supervisor must own its process group and session')
        return result


def process_snapshot(pid: int) -> tuple[ProcessIdentity, str] | None:
    if pid <= 0:
        raise ValueError('A process PID must be positive')
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm may contain spaces or parentheses. Fields after its final ')' start at state (3).
    fields = raw.rsplit(')', 1)[1].split()
    identity = ProcessIdentity(pid, int(fields[19]), int(fields[2]), int(fields[3]),
        Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    return identity, fields[0]


def capture_supervisor(pid: int) -> ProcessIdentity:
    snapshot = process_snapshot(pid)
    if snapshot is None:
        raise ProcessLookupError(pid)
    identity = ProcessIdentity.from_document(snapshot[0].document())
    descriptor = _pidfd_open(pid)
    try:
        _pidfd_signal(descriptor, 0)
    finally:
        os.close(descriptor)
    return identity


def owned_members(owner: ProcessIdentity) -> tuple[ProcessIdentity, ...]:
    """Exclude zombies and refuse a recycled PID or a different host boot."""
    if owner.boot_id != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return ()
    leader = process_snapshot(owner.pid)
    if leader is not None and leader[0] != owner:
        raise RuntimeError('Worker supervisor PID was reused; refusing to signal it')
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        snapshot = process_snapshot(int(entry.name))
        if snapshot is None:
            continue
        item, state = snapshot
        if state not in {'Z', 'X'} and item.group == owner.group and item.session == owner.session \
                and item.start_ticks >= owner.start_ticks:
            members.append(item)
    return tuple(members)


def signal_owned(owner: ProcessIdentity, number: signal.Signals) -> int:
    """Signal stable pidfds only; never send killpg to a saved, unverified PID."""
    count = 0
    for member in owned_members(owner):
        try:
            descriptor = _pidfd_open(member.pid)
        except ProcessLookupError:
            continue
        try:
            current = process_snapshot(member.pid)
            if current is not None and current[0] == member:
                _pidfd_signal(descriptor, number)
                count += 1
        except ProcessLookupError:
            pass
        finally:
            os.close(descriptor)
    return count
