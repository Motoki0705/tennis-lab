"""Safe filesystem access for local ops state and its locks."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Literal, TextIO
from uuid import uuid4

Entry = Literal["file", "directory", "legacy_temporary", "temporary"]


class StateFileError(OSError):
    def __init__(
        self,
        path: Path,
        reason: str,
        *,
        entry: Entry = "file",
        location: Path | None = None,
        error_number: int | None = None,
    ) -> None:
        super().__init__(f"{entry} {location or path}: {reason}")
        self.path = path
        self.entry = entry
        self.reason = reason
        self.errno = error_number


def _file_info(
    path: Path, directory: int, location: Path, entry: Entry = "file"
) -> os.stat_result | None:
    try:
        info = os.stat(location.name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _check_file(path, location, entry, info)
    return info


def _check_file(path: Path, location: Path, entry: Entry, info: os.stat_result) -> None:
    if stat.S_ISLNK(info.st_mode):
        reason = "symlink"
    elif not stat.S_ISREG(info.st_mode):
        reason = "not_regular"
    elif info.st_uid != os.geteuid():
        reason = "wrong_owner"
    elif info.st_nlink != 1:
        reason = "hardlink"
    elif info.st_mode & 0o022:
        reason = "shared_write"
    else:
        return
    raise StateFileError(path, reason, entry=entry, location=location)


@contextmanager
def _parent_fd(path: Path, *, create: bool = False) -> Iterator[int]:
    root, *parts = path.absolute().parent.parts
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, flags)
    location = Path(root)
    try:
        for name in parts:
            location /= name
            try:
                try:
                    child = os.open(name, flags, dir_fd=directory)
                except FileNotFoundError:
                    if not create:
                        raise
                    with suppress(FileExistsError):
                        os.mkdir(name, mode=0o700, dir_fd=directory)
                    child = os.open(name, flags, dir_fd=directory)
            except OSError as exc:
                if exc.errno not in {errno.ENOTDIR, errno.ELOOP}:
                    raise
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                reason = "symlink" if stat.S_ISLNK(info.st_mode) else "not_directory"
                raise StateFileError(
                    path, reason, entry="directory", location=location
                ) from exc
            os.close(directory)
            directory = child
        info = os.fstat(directory)
        if info.st_uid != os.geteuid():
            raise StateFileError(
                path, "wrong_owner", entry="directory", location=location
            )
        if info.st_mode & 0o022:
            raise StateFileError(
                path, "shared_write", entry="directory", location=location
            )
        yield directory
    finally:
        os.close(directory)


@contextmanager
def _open_file(path: Path, mode: Literal["r", "a"]) -> Iterator[TextIO]:
    with ExitStack() as opened:
        try:
            with _parent_fd(path, create=mode == "a") as directory:
                _file_info(path, directory, path)

                def opener(_path: str, flags: int) -> int:
                    return os.open(
                        path.name,
                        flags | os.O_NOFOLLOW | os.O_NONBLOCK,
                        0o600,
                        dir_fd=directory,
                    )

                stream = opened.enter_context(
                    open(path, mode, encoding="utf-8", opener=opener)
                )
                _check_file(path, path, "file", os.fstat(stream.fileno()))
                if mode == "a":
                    os.fchmod(stream.fileno(), 0o600)
        except (FileNotFoundError, StateFileError):
            raise
        except OSError as exc:
            reason = "symlink" if exc.errno == errno.ELOOP else "access_failed"
            raise StateFileError(path, reason, error_number=exc.errno) from exc
        yield stream


def read_state_text(path: Path) -> str:
    with _open_file(path, "r") as stream:
        try:
            return stream.read()
        except OSError as exc:
            raise StateFileError(path, "access_failed", error_number=exc.errno) from exc


def open_state_lock(path: Path) -> AbstractContextManager[TextIO]:
    return _open_file(path, "a")


def _same_inode(directory: int, name: str, created: os.stat_result) -> bool:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return os.path.samestat(created, info)


@contextmanager
def atomic_state_writer(path: Path) -> Iterator[TextIO]:
    """Commit a complete state file and preserve abandoned entries for inspection."""
    try:
        with _parent_fd(path, create=True) as directory:
            _file_info(path, directory, path)
            _file_info(
                path, directory, path.with_name(path.name + ".tmp"), "legacy_temporary"
            )
            prefix = f".{path.name}."
            for name in os.listdir(directory):
                if name.startswith(prefix) and name.endswith(".tmp"):
                    _file_info(path, directory, path.with_name(name), "temporary")

            temporary = path.with_name(f"{prefix}{uuid4().hex}.tmp")
            try:
                fd = os.open(
                    temporary.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError as exc:
                _file_info(path, directory, temporary, "temporary")
                raise StateFileError(
                    path, "temporary_collision", entry="temporary", location=temporary
                ) from exc

            created = None
            try:
                created = os.fstat(fd)
                with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
                    yield stream
                    stream.flush()
                    os.fsync(fd)
                # The pinned directory is owner-writable only. Recheck the
                # entries after serialization before committing under that fd.
                _file_info(path, directory, path)
                current = _file_info(path, directory, temporary, "temporary")
                if current is None or not os.path.samestat(created, current):
                    raise StateFileError(
                        path, "temporary_changed", entry="temporary", location=temporary
                    )
                os.replace(
                    temporary.name,
                    path.name,
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                )
                created = None
            finally:
                try:
                    if created is not None and _same_inode(
                        directory, temporary.name, created
                    ):
                        with suppress(FileNotFoundError):
                            os.unlink(temporary.name, dir_fd=directory)
                finally:
                    os.close(fd)
    except StateFileError:
        raise
    except OSError as exc:
        raise StateFileError(path, "write_failed", error_number=exc.errno) from exc
