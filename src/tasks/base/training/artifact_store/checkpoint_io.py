"""Lightning CheckpointIO that persists local checkpoints through an ArtifactStore."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from lightning_fabric.plugins.io import CheckpointIO, TorchCheckpointIO

from src.utils.artifact_store import ArtifactStore


class PublishingCheckpointIO(CheckpointIO):
    """Delegate checkpoint serialization locally, then atomically publish it."""

    def __init__(self, store: ArtifactStore) -> None:
        super().__init__()
        self.store = store
        self.local = TorchCheckpointIO()

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
        storage_options: Any | None = None,
    ) -> None:
        local_path = Path(path)
        self.local.save_checkpoint(checkpoint, local_path, storage_options)
        self.store.publish_file(local_path)

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: Any | None = None,
        weights_only: bool | None = None,
    ) -> dict[str, Any]:
        local_path = Path(path)
        if not local_path.exists():
            self.store.fetch_file(local_path)
        return cast(
            dict[str, Any],
            self.local.load_checkpoint(
                local_path,
                map_location=map_location,
                weights_only=weights_only,
            ),
        )

    def remove_checkpoint(self, path: str | Path) -> None:
        local_path = Path(path)
        self.local.remove_checkpoint(local_path)
        self.store.remove_file(local_path)
