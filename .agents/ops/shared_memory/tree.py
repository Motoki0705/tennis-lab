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
    def resolve(self, path: str) -> str: ...
    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None: ...
    def children(self, path: str) -> tuple[str, ...]: ...
    def read_bytes(self, path: str) -> bytes: ...


def _relative_parts(path: str) -> tuple[str, ...]:
    requested = PurePosixPath(path)
    if requested.is_absolute() or "\\" in path or "\0" in path:
        raise ValueError("path resolves outside the repository")
    return requested.parts


class Checkout:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _resolve(self, path: str, *, follow_links: bool = True) -> Path:
        resolved = self.root
        pending = deque(_relative_parts(path))
        symlinks = 0
        while pending:
            part = pending.popleft()
            if part == "..":
                if resolved == self.root:
                    raise ValueError("path resolves outside the repository")
                resolved = resolved.parent
                continue
            candidate = resolved / part
            if not pending and not follow_links:
                return candidate
            try:
                mode = candidate.lstat().st_mode
            except FileNotFoundError:
                if pending:
                    raise
                # A proposed generated document can be missing on disk.
                return candidate
            if not stat.S_ISLNK(mode):
                if pending and not stat.S_ISDIR(mode):
                    raise NotADirectoryError()
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

    def resolve(self, path: str) -> str:
        return self._resolve(path).relative_to(self.root).as_posix()

    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None:
        try:
            resolved = self._resolve(path, follow_links=follow_links)
            mode = resolved.stat(follow_symlinks=follow_links).st_mode
        except (FileNotFoundError, NotADirectoryError):
            return None
        if stat.S_ISLNK(mode):
            return "symlink"
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

    def resolve(self, path: str) -> str:
        resolved: list[str] = []
        parts = _relative_parts(path)
        for index, part in enumerate(parts):
            if part == "..":
                if not resolved:
                    raise ValueError("path resolves outside the repository")
                resolved.pop()
            else:
                resolved.append(part)
                if index + 1 < len(parts) and "/".join(resolved) not in self._children:
                    raise FileNotFoundError()
        return "/".join(resolved) or "."

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
