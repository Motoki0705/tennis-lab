"""Small planning metadata shared by Court nodes, never model instances."""

from pathlib import Path
from typing import Any

from src.tasks.court_detection.inference.checkpoint import (
    CourtCheckpointMetadata,
    read_court_checkpoint_metadata,
)
from src.tennis_scene.pipeline.model_assets import file_identity
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.models.loading.dinov3 import dinov3_source_assets


def _signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class CourtAssetCatalog:
    """One Court config shares CPU metadata and stat-keyed digests across nodes.

    A new configuration creates a fresh catalog. Changed files invalidate
    metadata; artifact-bound SHA checks still reject changes in an old plan.
    """

    def __init__(self) -> None:
        self._metadata: dict[tuple[Path, RuntimePathRoots], tuple[tuple[int, ...], CourtCheckpointMetadata]] = {}
        self._identities: dict[Path, tuple[tuple[int, ...], dict[str, Any]]] = {}
        self._sources: dict[Path, tuple[dict[Path, tuple[int, ...]], dict[str, Path]]] = {}

    def file_identity(self, path: Path) -> dict[str, Any]:
        signature = _signature(path)
        cached = self._identities.get(path)
        if cached is None or cached[0] != signature:
            identity = file_identity(path)
            if _signature(path) != signature:
                raise ValueError(f"Asset changed while binding identity: {path}")
            cached = signature, identity
            self._identities[path] = cached
        return dict(cached[1])

    def metadata(self, path: Path, resolver: PathResolver) -> CourtCheckpointMetadata:
        signature = _signature(path)
        key = path, resolver.roots
        cached = self._metadata.get(key)
        if cached is None or cached[0] != signature:
            self.file_identity(path)
            metadata = read_court_checkpoint_metadata(path, resolver=resolver)
            if _signature(path) != signature:
                raise ValueError(f"Court checkpoint changed while reading metadata: {path}")
            cached = signature, metadata
            self._metadata[key] = cached
        return cached[1]

    def paths(self, path: Path, resolver: PathResolver) -> dict[str, Path]:
        encoder = self.metadata(path, resolver).spec.model.encoder
        backbone, repository = encoder.checkpoint_path, encoder.repository_path
        if backbone is None or not backbone.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {backbone}")
        if repository is None:
            raise FileNotFoundError("DINOv3 repository is required")
        cached = self._sources.get(repository)
        if cached is None or any(not directory.is_dir() or _signature(directory) != signature for directory, signature in cached[0].items()):
            sources = dinov3_source_assets(repository)
            directories = (repository, *(entry for entry in repository.rglob("*") if entry.is_dir()))
            cached = {directory: _signature(directory) for directory in directories}, sources
            self._sources[repository] = cached
        return {"checkpoint": path, "backbone": backbone, **cached[1]}

    def identities(self, path: Path, resolver: PathResolver) -> dict[str, Any]:
        return {name: self.file_identity(asset) for name, asset in self.paths(path, resolver).items()}

    def checkpoint_identity(self, path: Path, resolver: PathResolver) -> dict[str, Any]:
        metadata = self.metadata(path, resolver)
        backbone = metadata.spec.model.encoder.checkpoint_path
        assert backbone is not None
        return metadata.identity(path, self.file_identity(path)["sha256"], self.file_identity(backbone)["sha256"])
