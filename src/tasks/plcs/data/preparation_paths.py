"""Explicit absolute paths for standalone PLCS version preparation."""

from pathlib import Path

from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def preparation_resolver(source: Path, destination: Path) -> PathResolver:
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("source and destination must be explicit absolute paths")
    return PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT,
            data_root=source.resolve().parent,
            artifact_root=destination.resolve().parent,
            output_root=destination.resolve().parent,
            checkpoint_root=PROJECT_ROOT,
            cache_root=PROJECT_ROOT,
            external_asset_root=PROJECT_ROOT,
        )
    )
