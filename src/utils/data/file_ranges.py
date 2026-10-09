"""Read exact immutable file ranges into one allocation without mmap faults."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.utils.shared_file_verification import FileVersion, file_version, stat_version


class FileRangeReader:
    """Bounded process-local descriptors; preadv copies only requested ranges.

    A trusted expected version must come from the caller's checksum verification.
    Random advice prevents broad read-ahead across unsampled JPEG frames. Bytes
    are copied directly into the final packed array, in the caller's exact order.
    """

    def __init__(self, max_open_files: int = 32) -> None:
        self._fds: OrderedDict[Path, int] = OrderedDict()
        self._pid = os.getpid()
        if max_open_files < 1:
            raise ValueError("Descriptor limit must be positive")
        self.max_open_files = max_open_files

    def _descriptor(self, path: Path) -> int:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        fd = self._fds.pop(path, None)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except BaseException:
                os.close(fd)
                raise
        self._fds[path] = fd
        while len(self._fds) > self.max_open_files:
            _, old_fd = self._fds.popitem(last=False)
            os.close(old_fd)
        return fd

    def read(self, path: Path, ranges: Sequence[tuple[int, int]], expected: FileVersion) -> NDArray[np.uint8]:
        if not path.is_absolute() or not ranges or any(offset < 0 or length < 1 or offset + length > expected[2]
                                                       for offset, length in ranges):
            raise ValueError("Expected absolute path and nonempty in-bounds file ranges")
        fd = self._descriptor(path)
        if stat_version(os.fstat(fd)) != expected or file_version(path) != expected:
            raise ValueError("File changed after checksum verification")
        packed: NDArray[np.uint8] = np.empty(sum(length for _, length in ranges), np.uint8)
        view = memoryview(packed)
        destination = 0
        for offset, length in ranges:
            received = 0
            while received < length:
                count = os.preadv(fd, [view[destination+received:destination+length]], offset+received)
                if count == 0:
                    raise ValueError("Unexpected EOF reading verified file range")
                received += count
            destination += length
        if stat_version(os.fstat(fd)) != expected or file_version(path) != expected:
            raise ValueError("File changed while reading verified ranges")
        return packed

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__, _fds=OrderedDict())

    def __del__(self) -> None:
        with suppress(OSError, AttributeError):
            self.close()
