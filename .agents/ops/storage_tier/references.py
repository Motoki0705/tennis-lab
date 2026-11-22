"""Validate the explicit unmanaged D: references without traversing them."""

from __future__ import annotations

import os
import stat
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from .filesystem import (
    StorageError,
    fd_mount_id,
    identity,
    real_directory_handle,
    stamp,
)

if TYPE_CHECKING:
    from .layout import Config


def data_parents(config: Config) -> set[str]:
    paths = (
        *config.containers,
        *config.datasets,
        *config.residents,
        *config.external_links,
    )
    return {
        "data",
        *config.containers,
        *(
            str(parent)
            for path in paths
            for parent in PurePosixPath(path).parents
            if str(parent).startswith("data")
        ),
    }


class ReferenceError(StorageError):
    def __init__(self, message: str, observations: list[dict[str, Any]]) -> None:
        self.observations = observations
        super().__init__(message)


def inspect_references(
    config: Config, *, read_targets: bool = True
) -> list[dict[str, Any]]:
    """Read each allowed link once, recording unavoidable Linux atime effects.

    Mount verification is the caller's responsibility and must happen first.
    Root migration symlinks remain layout errors without reading their targets.
    """
    root = config.repo / "data"
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            return []
    except FileNotFoundError:
        return []
    observations: list[dict[str, Any]] = []
    for relative_parent in sorted(
        data_parents(config), key=lambda value: (len(PurePosixPath(value).parts), value)
    ):
        parent = config.repo / relative_parent
        try:
            with real_directory_handle(parent, config.repo) as (parent_fd, _):
                with os.scandir(parent_fd) as entries:
                    children = sorted(entry.name for entry in entries)
                for name in children:
                    relative = relative_parent + "/" + name
                    info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    allowed = relative in config.external_links
                    if not stat.S_ISLNK(info.st_mode):
                        if allowed:
                            raise StorageError(
                                f"unmanaged link must be a symlink: {relative}"
                            )
                        continue
                    if not allowed:
                        raise StorageError(
                            f"undeclared unmanaged data link: {relative}"
                        )
                    if not read_targets:
                        observations.append(
                            {
                                "path": relative,
                                "expected_target": str(config.mirror / relative),
                                "verification": "deferred-explicit-dry-run",
                                "verified": False,
                                "method": "lstat-only",
                                "atime_before_ns": info.st_atime_ns,
                                "atime_after_ns": info.st_atime_ns,
                                "atime_changed": False,
                            }
                        )
                        continue
                    fd = os.open(name, os.O_PATH | os.O_NOFOLLOW, dir_fd=parent_fd)
                    try:
                        before = os.fstat(fd)
                        if (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
                            raise StorageError(
                                f"unmanaged link changed before reading: {relative}"
                            )
                        target = os.readlink("", dir_fd=fd)
                        after = os.fstat(fd)
                        record = {
                            "path": relative,
                            "target": target,
                            "expected_target": str(config.mirror / relative),
                            "atime_before_ns": before.st_atime_ns,
                            "atime_after_ns": after.st_atime_ns,
                            "atime_changed": before.st_atime_ns != after.st_atime_ns,
                            "method": "O_PATH|O_NOFOLLOW/readlinkat",
                        }
                        observations.append(record)
                        observed = os.stat(
                            name, dir_fd=parent_fd, follow_symlinks=False
                        )
                        if identity(stamp(before)) != identity(
                            stamp(after)
                        ) or identity(stamp(after)) != identity(stamp(observed)):
                            raise StorageError(
                                f"unmanaged link changed while reading: {relative}"
                            )
                        if target != str(config.mirror / relative):
                            raise StorageError(
                                f"unmanaged link target mismatch: {relative}"
                            )
                        expected = config.mirror / relative
                        with real_directory_handle(expected.parent, config.mount) as (
                            eparent,
                            mount_id,
                        ):
                            target_info = os.stat(
                                expected.name, dir_fd=eparent, follow_symlinks=False
                            )
                            if (
                                not (
                                    stat.S_ISDIR(target_info.st_mode)
                                    or stat.S_ISREG(target_info.st_mode)
                                )
                                or target_info.st_dev != os.fstat(eparent).st_dev
                            ):
                                raise StorageError(
                                    f"unmanaged link target must be a real E: file/directory: {relative}"
                                )
                            target_fd = os.open(
                                expected.name, os.O_PATH | os.O_NOFOLLOW, dir_fd=eparent
                            )
                            try:
                                pinned = os.fstat(target_fd)
                                if (pinned.st_dev, pinned.st_ino) != (
                                    target_info.st_dev,
                                    target_info.st_ino,
                                ) or fd_mount_id(target_fd) != mount_id:
                                    raise StorageError(
                                        f"unmanaged link target changed or crosses a mount: {relative}"
                                    )
                            finally:
                                os.close(target_fd)
                        record["verified"] = True
                        record["verification"] = "verified"
                    finally:
                        os.close(fd)
        except FileNotFoundError as exc:
            # Missing mixed containers are expected before hydration. Missing
            # targets after reading a link are not absent containers.
            if observations and observations[-1].get("verification") not in {
                "verified",
                "deferred-explicit-dry-run",
            }:
                raise ReferenceError(
                    f"unmanaged link target is missing: {observations[-1]['path']}",
                    observations,
                ) from exc
            try:
                parent.lstat()
            except FileNotFoundError:
                continue
            raise ReferenceError(
                f"unmanaged link path changed: {parent}", observations
            ) from exc
        except (OSError, StorageError) as exc:
            raise ReferenceError(str(exc), observations) from exc
    return observations
