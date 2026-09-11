"""No-op local artifact store."""

from __future__ import annotations

from pathlib import Path

from .contracts import ArtifactStore, ArtifactStoreError


class LocalArtifactStore(ArtifactStore):
    """Keep artifacts only in their runner-owned local directory."""

    @property
    def enabled(self) -> bool:
        return False

    def publish_file(self, path: Path) -> None:
        self.relative_path(path)
        if not path.is_file() or path.is_symlink():
            raise ArtifactStoreError(f"artifact is not a regular file: {path}")

    def fetch_file(self, path: Path) -> None:
        self.relative_path(path)
        if not path.is_file() or path.is_symlink():
            raise ArtifactStoreError(f"local artifact does not exist: {path}")

    def remove_file(self, path: Path) -> None:
        self.relative_path(path)

    def sync_tree(self) -> None:
        if not self.local_root.is_dir() or self.local_root.is_symlink():
            raise ArtifactStoreError(
                f"local artifact root is not a regular directory: {self.local_root}"
            )
