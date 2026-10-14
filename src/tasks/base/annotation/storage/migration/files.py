"""Read selected files without following symlinks, then copy verified bytes."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from ...artifacts import sync_directory
from ...contracts import relative_path
from ..contracts import ArtifactKind
from .contracts import BundleFile


def checked_root(root: Path) -> Path:
    if not root.is_absolute() or root != root.resolve():
        raise ValueError("migration root must be absolute, normalized and symlink-free")
    if not root.is_dir():
        raise ValueError("migration source root must be an existing directory")
    return root


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def read_file(root: Path, relative: str) -> Iterator[BinaryIO]:
    """Open from / through directory fds; reject replacement during the read."""
    checked_root(root)
    relative_path(relative)
    parts = (*root.parts[1:], *Path(relative).parts)
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            os.close(parent)
            parent = child
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent
        )
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("migration accepts only regular files")
            yield source
            after = os.fstat(source.fileno())
            current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if _signature(before) != _signature(after) or _signature(
                after
            ) != _signature(current):
                raise ValueError("migration source changed during reading")
    finally:
        os.close(parent)


def describe_file(
    root: Path,
    relative: str,
    *,
    kind: ArtifactKind,
    dependencies: tuple[str, ...] = (),
) -> BundleFile:
    digest = hashlib.sha256()
    size = 0
    with read_file(root, relative) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return BundleFile(
        path=relative,
        digest=digest.hexdigest(),
        size_bytes=size,
        kind=kind,
        dependencies=dependencies,
    )


def verify_file(root: Path, item: BundleFile) -> None:
    observed = describe_file(
        root, item.path, kind=item.kind, dependencies=item.dependencies
    )
    if observed != item:
        raise ValueError(f"migration file differs from dry-run: {item.path}")


def copy_file(source_root: Path, target_root: Path, item: BundleFile) -> None:
    """Never hardlink mutable legacy files or overwrite a conflicting target."""
    target = target_root / item.path
    for part in (target, *target.parents):
        if part.is_symlink():
            raise ValueError("migration destination must not contain symlinks")
        if part == target_root:
            break
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify_file(target_root, item)
        return
    temporary: Path | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        # Partials live outside the reader root and remain inside registered staging.
        partials = target_root.parent / "partials"
        if partials.is_symlink():
            raise ValueError("migration partial directory must not be a symlink")
        partials.mkdir(exist_ok=True)
        with (
            read_file(source_root, item.path) as source,
            tempfile.NamedTemporaryFile(dir=partials, delete=False) as output,
        ):
            temporary = Path(output.name)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                size += len(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != item.digest or size != item.size_bytes:
            raise ValueError(f"migration source differs from dry-run: {item.path}")
        # link() is an atomic no-clobber publication of our independent byte copy.
        os.link(temporary, target)
        sync_directory(target.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
