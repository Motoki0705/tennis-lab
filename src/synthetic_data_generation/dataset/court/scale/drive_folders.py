"""Attempt-local folder identities with fresh, duplicate-aware observations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any


class DriveFolders:
    """Cache positive IDs only; never cache a file or an absence decision.

    Known ancestor IDs allow their listings to run independently without path
    rediscovery. Every lookup still checks every link against new Drive listings,
    including the project root. A replacement or duplicate stops the attempt.
    """

    def __init__(self, session: Any, remote_root: str) -> None:
        self.session = session
        self.remote_name, root = remote_root.split(":", 1)
        self.root = PurePosixPath(root).parts
        self.ids: dict[tuple[str, ...], str] = {}
        self.pool = ThreadPoolExecutor(max_workers=8)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        self.ids.clear()

    def _fs(self, prefix: tuple[str, ...]) -> dict[str, str]:
        return {
            "_name": self.remote_name,
            "_root": "",
            **({"root_folder_id": self.ids[prefix]} if prefix else {}),
        }

    def _list(self, prefix: tuple[str, ...]) -> list[dict[str, Any]]:
        value = self.session.call(
            "operations/list",
            fs=self._fs(prefix),
            remote="",
            opt={"showHash": True, "noModTime": True, "noMimeType": True},
        ).get("list")
        if not isinstance(value, list):
            raise ValueError("Drive returned an unexpected folder listing.")
        return value

    @staticmethod
    def _match(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        matches = [v for v in items if v.get("Name", v.get("Path")) == name]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous Drive name: {name}; IDs={[v.get('ID') for v in matches]}"
            )
        if not matches:
            return None
        if not matches[0].get("ID"):
            raise ValueError("Drive did not return an entry ID.")
        return matches[0]

    def lookup(
        self, relative: str, *, create_parents: bool = False
    ) -> tuple[dict[str, str], dict[str, Any] | None] | None:
        components = (*self.root, *PurePosixPath(relative).parts)
        parents = []
        for i in range(len(components)):
            prefix = components[:i]
            if prefix and prefix not in self.ids:
                break
            parents.append(prefix)
        # All futures finish (including failures) before a caller can mutate Drive.
        snapshots = dict(zip(parents, self.pool.map(self._list, parents), strict=True))
        for i, name in enumerate(components):
            parent, prefix = components[:i], components[: i + 1]
            items = snapshots[parent] if parent in snapshots else self._list(parent)
            entry = self._match(items, name)
            if i == len(components) - 1:
                return self._fs(parent), entry
            if entry is None:
                if prefix in self.ids:
                    raise ValueError("Previously verified Drive folder disappeared.")
                if i < len(self.root):
                    raise ValueError("Drive project root does not exist.")
                if not create_parents:
                    return None
                self.session.call("operations/mkdir", fs=self._fs(parent), remote=name)
                entry = self._match(self._list(parent), name)
                if entry is None:
                    raise ValueError("Drive folder creation was not acknowledged.")
            if not entry.get("IsDir"):
                raise ValueError("Drive path ancestor is not a directory.")
            identifier = entry["ID"]
            if prefix in self.ids and self.ids[prefix] != identifier:
                raise ValueError("Previously verified Drive folder ID changed.")
            self.ids[prefix] = identifier
        raise AssertionError("Expected a nonempty Drive file path.")
