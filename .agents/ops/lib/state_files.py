"""Safe filesystem access for local ops state and its locks."""

from __future__ import annotations

import errno
import grp
import os
import pwd
import stat
from collections.abc import Callable, Iterator
from contextlib import (
    AbstractContextManager,
    ExitStack,
    contextmanager,
    nullcontext,
    suppress,
)
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StateDirectory:
    path: Path
    fd: int

    def verify(self, target: Path, *, anchor: StateDirectory | None = None) -> None:
        try:
            with open_state_directory(
                self.path, target=target, anchor=anchor
            ) as current:
                if os.path.samestat(os.fstat(self.fd), os.fstat(current.fd)):
                    return
        except FileNotFoundError:
            pass
        raise StateFileError(
            target, "directory_changed", entry="directory", location=self.path
        )

    def verify_lock(self, path: Path, stream: TextIO) -> None:
        info = _file_info(path, self.fd, path)
        if info is None or not os.path.samestat(info, os.fstat(stream.fileno())):
            raise StateFileError(path, "lock_changed")


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _is_user_private_group(info: os.stat_result, owner: int) -> bool:
    if info.st_uid != owner:
        return False
    try:
        account = pwd.getpwuid(owner)
        if account.pw_gid != info.st_gid:
            return False
        group = grp.getgrgid(info.st_gid)
        if any(member != account.pw_name for member in group.gr_mem):
            return False
        primary_users = {
            entry.pw_uid for entry in pwd.getpwall() if entry.pw_gid == info.st_gid
        }
    except (KeyError, OSError):
        return False
    # Enumeration has to include the owner to verify the group's membership.
    return primary_users == {owner}


def _check_directory(fd: int, path: Path, target: Path, *, leaf: bool) -> None:
    info = os.fstat(fd)
    owner = os.geteuid()
    if info.st_uid not in ({owner} if leaf else {0, owner}):
        raise StateFileError(target, "wrong_owner", entry="directory", location=path)
    shared_write = info.st_mode & 0o022
    if not shared_write:
        return
    if not leaf:
        # Sticky ancestors such as /tmp protect the next entry, whose owner
        # is checked in turn.
        if shared_write & 0o002 and info.st_mode & stat.S_ISVTX:
            return
        if shared_write == 0o020 and _is_user_private_group(info, owner):
            return
    raise StateFileError(target, "shared_write", entry="directory", location=path)


def _open_directory_at(
    parent: int, name: str, location: Path, target: Path, *, create: bool, new: bool
) -> int:
    try:
        if new:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        try:
            return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            if not create or new:
                raise
            with suppress(FileExistsError):
                os.mkdir(name, mode=0o700, dir_fd=parent)
            return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
    except OSError as exc:
        if exc.errno not in {errno.ENOTDIR, errno.ELOOP, errno.EEXIST}:
            raise
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            reason = "symlink"
        elif exc.errno == errno.EEXIST:
            raise
        else:
            reason = "not_directory"
        raise StateFileError(
            target, reason, entry="directory", location=location
        ) from exc


@contextmanager
def open_state_directory(
    path: Path,
    *,
    target: Path | None = None,
    anchor: StateDirectory | None = None,
    create: bool = False,
    new: bool = False,
) -> Iterator[StateDirectory]:
    """Open private state through trusted, no-follow directory entries."""
    path = path.absolute()
    target = target or path
    if anchor is None:
        root, *parts = path.parts
        location = Path(root)
        directory = os.open(root, _DIRECTORY_FLAGS)
    else:
        parts = list(path.relative_to(anchor.path).parts)
        if ".." in parts:
            raise ValueError("state path escapes the pinned directory")
        location = anchor.path
        directory = os.dup(anchor.fd)
    try:
        _check_directory(directory, location, target, leaf=not parts)
        for position, name in enumerate(parts, 1):
            location /= name
            leaf = position == len(parts)
            child = _open_directory_at(
                directory, name, location, target, create=create, new=new and leaf
            )
            os.close(directory)
            directory = child
            _check_directory(directory, location, target, leaf=leaf)
        if new and not parts:
            raise FileExistsError(path)
        yield StateDirectory(path, directory)
    finally:
        os.close(directory)


def create_state_directory(
    path: Path, *, target: Path | None = None, parents: bool = False
) -> None:
    with open_state_directory(path, target=target, create=parents, new=True):
        pass


def _file_directory(
    path: Path, directory: StateDirectory | None, *, create: bool
) -> AbstractContextManager[StateDirectory]:
    if directory is None:
        return open_state_directory(path.parent, target=path, create=create)
    if path.parent.absolute() != directory.path:
        raise ValueError("state file must be inside its pinned directory")
    return nullcontext(directory)


@contextmanager
def _open_file(
    path: Path, mode: Literal["r", "a"], directory: StateDirectory | None
) -> Iterator[TextIO]:
    with ExitStack() as opened:
        try:
            with _file_directory(path, directory, create=mode == "a") as parent:
                _file_info(path, parent.fd, path)

                def opener(_path: str, flags: int) -> int:
                    return os.open(
                        path.name,
                        flags | os.O_NOFOLLOW | os.O_NONBLOCK,
                        0o600,
                        dir_fd=parent.fd,
                    )

                stream = opened.enter_context(
                    open(path, mode, encoding="utf-8", newline="", opener=opener)
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


def read_state_text(path: Path, *, directory: StateDirectory | None = None) -> str:
    with _open_file(path, "r", directory) as stream:
        try:
            return stream.read()
        except OSError as exc:
            raise StateFileError(path, "access_failed", error_number=exc.errno) from exc


def open_state_lock(
    path: Path, *, directory: StateDirectory | None = None
) -> AbstractContextManager[TextIO]:
    return _open_file(path, "a", directory)


def _same_inode(directory: int, name: str, created: os.stat_result) -> bool:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return os.path.samestat(created, info)


@contextmanager
def _temporary_writer(
    path: Path,
    directory: StateDirectory | None,
) -> Iterator[tuple[TextIO, PreparedStateWrite]]:
    try:
        with _file_directory(path, directory, create=True) as parent:
            directory_fd = parent.fd
            _file_info(path, directory_fd, path)
            _file_info(
                path,
                directory_fd,
                path.with_name(path.name + ".tmp"),
                "legacy_temporary",
            )
            prefix = f".{path.name}."
            for name in os.listdir(directory_fd):
                if name.startswith(prefix) and name.endswith(".tmp"):
                    _file_info(path, directory_fd, path.with_name(name), "temporary")

            temporary = path.with_name(f"{prefix}{uuid4().hex}.tmp")
            try:
                fd = os.open(
                    temporary.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError as exc:
                _file_info(path, directory_fd, temporary, "temporary")
                raise StateFileError(
                    path, "temporary_collision", entry="temporary", location=temporary
                ) from exc

            created = None
            try:
                created = os.fstat(fd)
                with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
                    yield (
                        stream,
                        PreparedStateWrite(path, temporary, directory_fd, created),
                    )
            finally:
                try:
                    if created is not None and _same_inode(
                        directory_fd, temporary.name, created
                    ):
                        with suppress(FileNotFoundError):
                            os.unlink(temporary.name, dir_fd=directory_fd)
                finally:
                    os.close(fd)
    except StateFileError:
        raise
    except OSError as exc:
        raise StateFileError(path, "write_failed", error_number=exc.errno) from exc


@dataclass(frozen=True)
class PreparedStateWrite:
    path: Path
    temporary: Path
    directory_fd: int
    created: os.stat_result

    def publish(self) -> None:
        _file_info(self.path, self.directory_fd, self.path)
        current = _file_info(self.path, self.directory_fd, self.temporary, "temporary")
        if current is None or not os.path.samestat(self.created, current):
            raise StateFileError(
                self.path,
                "temporary_changed",
                entry="temporary",
                location=self.temporary,
            )
        try:
            os.replace(
                self.temporary.name,
                self.path.name,
                src_dir_fd=self.directory_fd,
                dst_dir_fd=self.directory_fd,
            )
        except OSError as exc:
            raise StateFileError(
                self.path, "write_failed", error_number=exc.errno
            ) from exc

    def restore(self, previous: PreparedStateWrite | None) -> None:
        if not _same_inode(self.directory_fd, self.path.name, self.created):
            raise StateFileError(self.path, "file_changed")
        if previous is None:
            os.unlink(self.path.name, dir_fd=self.directory_fd)
        else:
            previous.publish()


@contextmanager
def prepare_state_write(
    path: Path,
    write: Callable[[TextIO], object],
    *,
    directory: StateDirectory | None = None,
) -> Iterator[PreparedStateWrite]:
    with _temporary_writer(path, directory) as (stream, prepared):
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
        yield prepared


@contextmanager
def atomic_state_writer(
    path: Path,
    *,
    directory: StateDirectory | None = None,
) -> Iterator[TextIO]:
    """Commit a complete state file and preserve abandoned entries for inspection."""
    with _temporary_writer(path, directory) as (stream, prepared):
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        prepared.publish()
