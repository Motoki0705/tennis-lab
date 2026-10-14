"""Bounded inode inventory; follow neither symlinks nor changing directory paths."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Literal

from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from .contracts import Inventory, InventoryEntry, InventoryScope


def scope_path(layout: AnnotationLayout, scope: InventoryScope) -> Path:
    return layout.artifact_path(
        ArtifactRef(role=scope.role, path=scope.path, digest="0" * 64, size_bytes=0)
    )


def inventory(
    layout: AnnotationLayout, scopes: tuple[InventoryScope, ...], *, max_entries: int
) -> Inventory:
    if not scopes or max_entries < 1:
        raise ValueError(
            "inventory requires explicit scopes and a positive entry budget"
        )
    roots = [scope_path(layout, scope) for scope in scopes]
    for index, root in enumerate(roots):
        if any(
            root.is_relative_to(other) or other.is_relative_to(root)
            for other in roots[:index]
        ):
            raise ValueError(
                "inventory scopes must not overlap, including role aliases"
            )
    entries: list[InventoryEntry] = []

    def append(path: Path, metadata: os.stat_result) -> None:
        if len(entries) >= max_entries:
            raise ValueError(
                "inventory entry budget exceeded; no partial report is returned"
            )
        mode = metadata.st_mode
        kind: Literal["file", "directory", "symlink", "special"] = (
            "file"
            if stat.S_ISREG(mode)
            else "directory"
            if stat.S_ISDIR(mode)
            else "symlink"
            if stat.S_ISLNK(mode)
            else "special"
        )
        entries.append(
            InventoryEntry(
                path=str(path),
                kind=kind,
                device=metadata.st_dev,
                inode=metadata.st_ino,
                links=metadata.st_nlink,
                logical_bytes=metadata.st_size,
                allocated_bytes=metadata.st_blocks * 512,
                modified_ns=metadata.st_mtime_ns,
                changed_ns=metadata.st_ctime_ns,
            )
        )

    def visit(path: Path, descriptor: int) -> None:
        before = os.fstat(descriptor)
        with os.scandir(descriptor) as iterator:
            names = sorted(entry.name for entry in iterator)
        for name in names:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child = path / name
            append(child, metadata)
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    current = os.fstat(child_fd)
                    if (metadata.st_dev, metadata.st_ino) != (
                        current.st_dev,
                        current.st_ino,
                    ):
                        raise ValueError("directory changed during inventory")
                    visit(child, child_fd)
                finally:
                    os.close(child_fd)
        after = os.fstat(descriptor)
        if (before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("directory changed during inventory")

    for root in roots:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            append(root, os.fstat(descriptor))
            visit(root, descriptor)
        finally:
            os.close(descriptor)
    unique = {(item.device, item.inode): item for item in entries}
    return Inventory(
        created_at_ns=time.time_ns(),
        scopes=scopes,
        entries=tuple(sorted(entries, key=lambda item: item.path)),
        logical_bytes=sum(item.logical_bytes for item in entries),
        allocated_bytes=sum(item.allocated_bytes for item in unique.values()),
        unique_inodes=len(unique),
        files=sum(item.kind == "file" for item in entries),
        symlinks=sum(item.kind == "symlink" for item in entries),
        hardlinked_files=sum(
            item.kind == "file" and item.links > 1 for item in entries
        ),
    )
