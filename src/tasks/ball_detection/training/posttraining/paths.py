"""Role-aware roots for explicit absolute research CLI arguments."""
from __future__ import annotations

import os
from pathlib import Path

from src.utils.configuration import PathResolver, RuntimePathRoots


def resolver(project: Path, artifacts: tuple[Path, ...], outputs: tuple[Path, ...], cache: Path) -> PathResolver:
    if not all(path.is_absolute() for path in (project, *artifacts, *outputs, cache)):
        raise ValueError("Research CLI paths must be explicit absolute paths")
    artifact_root = Path(os.path.commonpath([path.parent for path in artifacts]))
    output_root = Path(os.path.commonpath([path.parent for path in outputs]))
    return PathResolver(RuntimePathRoots(project_root=project, data_root=artifact_root, artifact_root=artifact_root,
        output_root=output_root, checkpoint_root=output_root, cache_root=cache, external_asset_root=project))
