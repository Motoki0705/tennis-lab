"""Notification-based driving of real annotation workers and service stages."""

from __future__ import annotations

import ctypes
import os
import select
import subprocess
from pathlib import Path


class AnnotationEvents:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify unavailable")
        self.watched: set[Path] = set()
        self.exited: set[int] = set()
        self.refresh()

    def refresh(self) -> None:
        for path in (self.root, *(p for p in self.root.rglob("*") if p.is_dir())):
            if path in self.watched:
                continue
            if (
                self.libc.inotify_add_watch(
                    self.fd, os.fsencode(path), 0x8 | 0x80 | 0x100
                )
                < 0
            ):
                raise OSError(ctypes.get_errno(), "inotify watch failed")
            self.watched.add(path)

    def wait(self, children: list[subprocess.Popen[bytes]], timeout: float) -> None:
        descriptors = []
        try:
            for child in children:
                if child.poll() is not None:
                    if child.pid not in self.exited:
                        self.exited.add(child.pid)
                        return
                    continue
                try:
                    descriptors.append(os.pidfd_open(child.pid))
                except ProcessLookupError:
                    return
            ready, _, _ = select.select([self.fd, *descriptors], [], [], timeout)
            if not ready:
                raise AssertionError(
                    "annotation produced no file or process notification before deadline"
                )
            if self.fd in ready:
                while True:
                    try:
                        os.read(self.fd, 65536)
                    except BlockingIOError:
                        break
        finally:
            for fd in descriptors:
                os.close(fd)

    def close(self) -> None:
        os.close(self.fd)
