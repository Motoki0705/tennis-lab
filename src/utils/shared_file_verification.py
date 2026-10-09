"""Process-shared verification of a fixed file set, including spawned workers."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .checksum import FileIntegrityError, dual_sha256

FileVersion = tuple[int, int, int, int, int]


def file_version(path: Path) -> FileVersion:
    return stat_version(path.stat())


def stat_version(stat: os.stat_result) -> FileVersion:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class SharedFileVerification:
    """Hash once per immutable file across workers; fail on later stat changes.

    Construct before starting workers. Only successful dual-provider checks are
    published. Small striped locks allow different files to be hashed in parallel.
    No server process or on-disk trust cache is created. The cache lives exactly
    as long as this dataset and its workers; every new run verifies independently.
    """

    def __init__(self, files: dict[Path, str]) -> None:
        self.paths = tuple(files)
        self.expected = tuple(files.values())
        self.indices = {path: i for i, path in enumerate(self.paths)}
        if not files or any(not p.is_absolute() for p in files):
            raise ValueError("Verification requires a nonempty absolute file set")
        context = mp.get_context("spawn")
        self._versions = context.RawArray("q", 6 * len(files))
        self._locks = [context.Lock() for _ in range(min(32, len(files)))]

    def verify(self, path: Path) -> FileVersion:
        index = self.indices[path]
        offset = 6 * index
        with self._locks[index % len(self._locks)]:
            version = file_version(path)
            if self._versions[offset]:
                if tuple(self._versions[offset + 1:offset + 6]) != version:
                    raise ValueError("Verified file changed during dataset use")
            else:
                if dual_sha256(path) != self.expected[index]:
                    raise ValueError("Frozen file checksum mismatch")
                if file_version(path) != version:
                    raise ValueError("File changed during verification")
                self._versions[offset + 1:offset + 6] = version
                self._versions[offset] = 1
        return version

    def verify_all(self, workers: int = 1, *, paths: Sequence[Path] | None = None,
                   progress: Callable[[int, int], None] | None = None) -> None:
        if workers < 1:
            raise ValueError("Verification worker count must be positive")
        indices = [self.indices[path] for path in (self.paths if paths is None else paths)]
        if not indices:
            return
        if workers == 1:
            for count, index in enumerate(indices, 1):
                self.verify(self.paths[index])
                if progress is not None:
                    progress(count, len(indices))
            return
        # Pass shared primitives only at process construction, never as task
        # payloads. Spawn is safe even if the parent has initialized CUDA.
        with ProcessPoolExecutor(max_workers=min(workers, len(indices)),
                                 mp_context=mp.get_context("spawn"),
                                 initializer=_initialize_verifier, initargs=(self,)) as pool:
            for count, _ in enumerate(pool.map(_verify_index, indices), 1):
                if progress is not None:
                    progress(count, len(indices))


_PROCESS_VERIFIER: SharedFileVerification | None = None


def _initialize_verifier(verifier: SharedFileVerification) -> None:
    global _PROCESS_VERIFIER
    _PROCESS_VERIFIER = verifier


def _verify_index(index: int) -> None:
    if _PROCESS_VERIFIER is None:
        raise RuntimeError("Verification worker has not been initialized")
    try:
        _PROCESS_VERIFIER.verify(_PROCESS_VERIFIER.paths[index])
    except FileIntegrityError as error:
        # FileIntegrityError requires keyword-only diagnostic data and cannot
        # use Exception's default pickle reconstruction across process pipes.
        raise ValueError(f"File integrity verification failed: {error}") from error
