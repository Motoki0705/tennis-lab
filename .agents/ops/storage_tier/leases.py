"""Linux inode write leases; break notifications stay on the calling thread."""

from __future__ import annotations

import fcntl
import signal
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .filesystem import StorageError

# Linux UAPI constants not exposed by Python 3.11's fcntl module.
F_SETOWN_EX = 15
F_OWNER_TID = 0


@contextmanager
def write_lease(fd: int) -> Iterator[None]:
    """Never fall back to unlocked deletion; release and drain on every exit."""
    if Path("/proc/sys/fs/leases-enable").read_text().strip() != "1":
        raise StorageError("exclusive file leases are disabled")
    signals = {signal.SIGIO}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    acquired = False
    try:
        # Process-wide SIGIO ownership can kill another unblocked thread. The
        # Linux extended owner explicitly directs this fd's SIGIO to our TID.
        fcntl.fcntl(
            fd, F_SETOWN_EX, struct.pack("ii", F_OWNER_TID, threading.get_native_id())
        )
        fcntl.fcntl(fd, fcntl.F_SETSIG, signal.SIGIO)
        try:
            fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_WRLCK)
        except OSError as exc:
            raise StorageError(f"exclusive file lease unavailable: {exc}") from exc
        acquired = True
        yield
    finally:
        final_state: int | None = None
        notifications = 0
        try:
            if acquired:
                try:
                    # Nothing that can block on fsync/journaling belongs between
                    # this last state observation and the unlock syscall.
                    final_state = fcntl.fcntl(fd, fcntl.F_GETLEASE)
                finally:
                    fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_UNLCK)
        finally:
            # Our thread owns these notifications. Consume them before restoring
            # its mask, but never discard evidence from the last check/unlock gap.
            try:
                while signal.sigtimedwait(signals, 0) is not None:
                    notifications += 1
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            if acquired and (
                (final_state is not None and final_state != fcntl.F_WRLCK)
                or notifications
            ):
                raise StorageError(
                    "exclusive file lease break during release: "
                    f"state={final_state}, SIGIO={notifications}"
                )


def require_write_lease(fd: int) -> int:
    value = fcntl.fcntl(fd, fcntl.F_GETLEASE)
    if value != fcntl.F_WRLCK:
        raise StorageError(f"exclusive file lease break pending or lost: {value}")
    return value
