"""Copy immutable CLI preparation without sharing writable files or git objects."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirectorySeed:
    original: Path
    snapshot: Path

    @classmethod
    def capture(cls, original: Path, snapshot: Path) -> DirectorySeed:
        if snapshot.is_relative_to(original):
            raise ValueError("A seed snapshot must be outside the source directory")
        shutil.copytree(original, snapshot, symlinks=True)
        return cls(original, snapshot)

    def restore(self, destination: Path) -> None:
        if any(
            destination.is_relative_to(root) or root.is_relative_to(destination)
            for root in (self.original, self.snapshot)
        ):
            raise ValueError("A seed must be restored into a separate private directory")
        if destination.exists():
            shutil.rmtree(destination)
        # copy2/copytree never hardlink the objects, refs, indexes or state files.
        shutil.copytree(self.snapshot, destination, symlinks=True)
        before, after = str(self.original).encode(), str(destination).encode()
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Seed contains an unsupported symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(destination)
            if relative.parts[:3] == ("repo", ".git", "objects"):
                continue
            payload = path.read_bytes()
            if before in payload:
                if b"\0" in payload:
                    raise ValueError(f"Seed contains a private path in binary state: {path}")
                path.write_bytes(payload.replace(before, after))
