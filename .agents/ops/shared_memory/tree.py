"""Read-only views over a checkout or a repo-relative file map."""

from __future__ import annotations

import stat
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

FileMap = Mapping[str, str | bytes]
Edits = Mapping[str, bytes | None]
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


def resolve_links(
    path: str,
    kind: Callable[[str], FileKind | None],
    read_link: Callable[[str], str],
    *,
    follow_links: bool = True,
    checkout_root: Path | None = None,
) -> str:
    """Resolve against repository nodes, allowing a missing leaf for generated files."""
    resolved = PurePosixPath(".")
    pending = deque(_relative_parts(path))
    symlinks = 0
    while pending:
        part = pending.popleft()
        if part == "..":
            if not resolved.parts:
                raise ValueError("path resolves outside the repository")
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if not pending and not follow_links:
            return candidate.as_posix()
        node = kind(candidate.as_posix())
        if node != "symlink":
            if pending and node is None:
                raise FileNotFoundError()
            if pending and node != "directory":
                raise NotADirectoryError()
            resolved = candidate
            continue
        symlinks += 1
        if symlinks > 40:  # Linux's path resolution limit
            raise ValueError("path exceeds the symlink resolution limit")
        target = PurePosixPath(read_link(candidate.as_posix()))
        if target.is_absolute():
            if checkout_root is None or not target.is_relative_to(checkout_root):
                raise ValueError("path resolves outside the repository")
            target = target.relative_to(checkout_root)
            resolved = PurePosixPath(".")
        pending.extendleft(reversed(_relative_parts(target.as_posix())))
    return resolved.as_posix()


class Checkout:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _kind(self, path: str) -> FileKind | None:
        try:
            mode = (self.root / path).lstat().st_mode
        except (FileNotFoundError, NotADirectoryError):
            return None
        if stat.S_ISLNK(mode):
            return "symlink"
        if stat.S_ISDIR(mode):
            return "directory"
        if stat.S_ISREG(mode):
            return "file"
        return "other"

    def _resolve(self, path: str, *, follow_links: bool = True) -> str:
        return resolve_links(
            path,
            self._kind,
            lambda relative: (self.root / relative).readlink().as_posix(),
            follow_links=follow_links,
            checkout_root=self.root,
        )

    def resolve(self, path: str) -> str:
        return self._resolve(path)

    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None:
        try:
            return self._kind(self._resolve(path, follow_links=follow_links))
        except (FileNotFoundError, NotADirectoryError):
            return None

    def children(self, path: str) -> tuple[str, ...]:
        directory = self.root / self.resolve(path)
        return tuple(sorted(child.name for child in directory.iterdir()))

    def read_bytes(self, path: str) -> bytes:
        return (self.root / self.resolve(path)).read_bytes()


class EditedTree:
    """Overlay regular file additions, replacements and deletions in existing dirs."""

    def __init__(self, tree: Tree, edits: Edits) -> None:
        self.tree = tree
        self.edits = edits

    def resolve(self, path: str) -> str:
        return self.tree.resolve(path)

    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None:
        if follow_links:
            try:
                path = self.resolve(path)
            except (FileNotFoundError, NotADirectoryError):
                return None
        if path in self.edits:
            return None if self.edits[path] is None else "file"
        return self.tree.kind(path, follow_links=follow_links)

    def children(self, path: str) -> tuple[str, ...]:
        names = set(self.tree.children(path))
        for name, content in self.edits.items():
            directory, _, filename = name.rpartition("/")
            if directory == path:
                if content is None:
                    names.discard(filename)
                else:
                    names.add(filename)
        return tuple(sorted(names))

    def read_bytes(self, path: str) -> bytes:
        path = self.resolve(path)
        if path not in self.edits:
            return self.tree.read_bytes(path)
        content = self.edits[path]
        if content is None:
            raise FileNotFoundError(path)
        return content


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
