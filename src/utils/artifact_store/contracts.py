"""Storage contract for publishing a local artifact tree."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStoreError(RuntimeError):
    """Raised when durable artifact persistence cannot be completed safely."""


class ArtifactStore(ABC):
    """Publish one explicitly bounded local tree to durable storage."""

    def __init__(self, local_root: Path) -> None:
        self.local_root = local_root.resolve(strict=False)

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this store persists outside the local filesystem."""

    @abstractmethod
    def publish_file(self, path: Path) -> None:
        """Atomically publish one regular file below ``local_root``."""

    @abstractmethod
    def fetch_file(self, path: Path) -> None:
        """Fetch one previously published file into ``path``."""

    @abstractmethod
    def remove_file(self, path: Path) -> None:
        """Remove one published file corresponding to ``path``."""

    @abstractmethod
    def sync_tree(self) -> None:
        """Copy the current local tree to durable storage without deleting remote data."""

    def relative_path(self, path: Path) -> Path:
        """Return a local-root-relative path, failing on boundary escape."""
        resolved = path.resolve(strict=False)
        try:
            relative = resolved.relative_to(self.local_root)
        except ValueError as error:
            raise ArtifactStoreError(
                f"artifact path is outside local_root {self.local_root}: {path}"
            ) from error
        if relative == Path("."):
            raise ArtifactStoreError("artifact operation requires a child path")
        return relative
