"""Read-only views over a checkout or a repo-relative file map."""

from __future__ import annotations

import stat
from collections import deque
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

FileMap = Mapping[str, str | bytes]
FileKind = Literal["directory", "file", "symlink", "other"]


class Tree(Protocol):
    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None: ...
    def children(self, path: str) -> tuple[str, ...]: ...
    def read_bytes(self, path: str) -> bytes: ...


class Checkout:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve(self, path: str) -> Path:
        requested = Path(path)
        if requested.is_absolute():
            raise ValueError("path resolves outside the repository")
        resolved = self.root
        pending = deque(requested.parts)
        symlinks = 0
        while pending:
            part = pending.popleft()
            if part == "..":
                if resolved == self.root:
                    raise ValueError("path resolves outside the repository")
                resolved = resolved.parent
                continue
            candidate = resolved / part
            if not candidate.is_symlink():
                resolved = candidate
                continue
            symlinks += 1
            if symlinks > 40:  # Linux's path resolution limit
                raise ValueError("path exceeds the symlink resolution limit")
            target = candidate.readlink()
            if target.is_absolute():
                if not target.is_relative_to(self.root):
                    raise ValueError("path resolves outside the repository")
                target = target.relative_to(self.root)
                resolved = self.root
            pending.extendleft(reversed(target.parts))
        return resolved

    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None:
        candidate = self.root / path
        if not follow_links and candidate.is_symlink():
            return "symlink"
        try:
            mode = self._resolve(path).stat().st_mode
        except (FileNotFoundError, NotADirectoryError):
            return None
        if stat.S_ISDIR(mode):
            return "directory"
        if stat.S_ISREG(mode):
            return "file"
        return "other"

    def children(self, path: str) -> tuple[str, ...]:
        return tuple(sorted(child.name for child in self._resolve(path).iterdir()))

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()


class MappedTree:
    """Map normalized POSIX paths to regular-file content and infer directories."""

    def __init__(self, files: FileMap) -> None:
        self._files = files
        self._children: dict[str, set[str]] = {}
        for path in files:
            node = PurePosixPath(path)
            if (
                node.is_absolute()
                or ".." in node.parts
                or node.as_posix() != path
                or not node.parts
            ):
                raise ValueError("file map keys must be normalized repo-relative paths")
            for parent in node.parents:
                self._children.setdefault(parent.as_posix(), set()).add(node.name)
                node = parent
        if self._children.keys() & files.keys():
            raise ValueError("file map includes a file with children")

    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None:
        if path in self._files:
            return "file"
        if path in self._children:
            return "directory"
        return None

    def children(self, path: str) -> tuple[str, ...]:
        return tuple(sorted(self._children[path]))

    def read_bytes(self, path: str) -> bytes:
        value = self._files[path]
        return value.encode("utf-8") if isinstance(value, str) else value
