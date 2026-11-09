"""Directory-fd filesystem access; no path component may follow a symlink."""

from __future__ import annotations

import errno
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .schema import ContractError


def relative_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or any(char in value for char in "%?#:")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ContractError("unsafe_path")
    return value


def asset_path(asset_id: str, value: str) -> str:
    relative_path(value)
    if not re.fullmatch(rf"assets/{re.escape(asset_id)}\.[a-z0-9]{{1,10}}", value):
        raise ContractError("unsafe_path")
    return value


def _open_directory(parent: int, name: str, *, create: bool) -> int:
    try:
        if create:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent)
                os.fsync(parent)
            except FileExistsError:
                pass
        return os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ContractError("unsafe_path") from exc
        raise


class SafeDirectory:
    """POSIX-only, including WSL. Keep the descriptor pinned during each operation."""

    def __init__(self, path: Path, *, create: bool = False):
        absolute = path.absolute()
        if ".." in absolute.parts:
            raise ContractError("unsafe_path")
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in absolute.parts[1:]:
                child = _open_directory(descriptor, part, create=create)
                os.close(descriptor)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
        self.fd = descriptor

    def __enter__(self) -> SafeDirectory:
        return self

    def __exit__(self, *_: object) -> None:
        os.close(self.fd)

    @contextmanager
    def _parent(self, path: str, *, create: bool = False) -> Iterator[tuple[int, str]]:
        parts = relative_path(path).split("/")
        descriptor = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                child = _open_directory(descriptor, part, create=create)
                os.close(descriptor)
                descriptor = child
            yield descriptor, parts[-1]
        finally:
            os.close(descriptor)

    def mkdir(self, path: str) -> None:
        with self._parent(path, create=True) as (parent, name):
            descriptor = _open_directory(parent, name, create=True)
            os.close(descriptor)

    def exists(self, path: str) -> bool:
        try:
            with self._parent(path) as (parent, name):
                result = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISLNK(result.st_mode):
                    raise ContractError("unsafe_path")
                return True
        except FileNotFoundError:
            return False

    @contextmanager
    def open_file(
        self, path: str, flags: int, *, create_parents: bool = False
    ) -> Iterator[int]:
        with self._parent(path, create=create_parents) as (parent, name):
            try:
                descriptor = os.open(
                    name,
                    flags | os.O_NOFOLLOW | os.O_NONBLOCK,
                    mode=0o600,
                    dir_fd=parent,
                )
            except OSError as exc:
                is_link = exc.errno == errno.ELOOP
                if exc.errno == errno.EEXIST:
                    is_link = stat.S_ISLNK(
                        os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
                    )
                if is_link:
                    raise ContractError("unsafe_path") from exc
                raise
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ContractError("unsafe_path")
                yield descriptor
            finally:
                os.close(descriptor)

    def read(self, path: str) -> bytes:
        with (
            self.open_file(path, os.O_RDONLY) as descriptor,
            os.fdopen(os.dup(descriptor), "rb") as handle,
        ):
            return handle.read()

    def write(self, path: str, data: bytes) -> None:
        with (
            self.open_file(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, create_parents=True
            ) as descriptor,
            os.fdopen(os.dup(descriptor), "wb") as handle,
        ):
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with self._parent(path) as (parent, _):
            os.fsync(parent)

    def rename(self, source: str, destination: str) -> None:
        if not self.exists(source):
            raise FileNotFoundError(source)
        with (
            self._parent(source) as (source_fd, source_name),
            self._parent(destination, create=True) as (
                destination_fd,
                destination_name,
            ),
        ):
            # Reject even a destination symlink that replace() would unlink.
            if self.exists(destination):
                raise ContractError("unsafe_path")
            os.rename(
                source_name,
                destination_name,
                src_dir_fd=source_fd,
                dst_dir_fd=destination_fd,
            )
            os.fsync(source_fd)
            os.fsync(destination_fd)

    def atomic_write(self, path: str, data: bytes) -> None:
        relative_path(path)
        parent_name, _, _ = path.rpartition("/")
        temporary = f"{parent_name + '/' if parent_name else ''}.pending-{uuid4().hex}"
        self.write(temporary, data)
        with self._parent(path) as (parent, name):
            self.exists(path)  # Refuse symlinks rather than silently replacing them.
            temporary_name = temporary.rsplit("/", 1)[-1]
            os.replace(temporary_name, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
