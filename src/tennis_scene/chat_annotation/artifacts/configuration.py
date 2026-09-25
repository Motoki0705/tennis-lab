"""Path authority for the explicitly selected local annotation workspace."""

from pathlib import Path

from src.utils.configuration.paths import PathResolver, RuntimePathRoots


def artifact_path_resolver(root: Path) -> PathResolver:
    """Scope all CLI path validation to one caller-selected output directory."""
    return PathResolver(
        RuntimePathRoots(
            project_root=root,
            data_root=root,
            checkpoint_root=root,
            artifact_root=root,
            output_root=root,
            cache_root=root,
            external_asset_root=root,
        )
    )
