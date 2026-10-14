"""Role roots for explicit experiment command-line path validation."""

from pathlib import Path

from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def experiment_paths(
    *,
    data_root: Path,
    output_root: Path,
    checkpoint: Path | None = None,
    artifact_root: Path | None = None,
) -> PathResolver:
    return PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT,
            data_root=data_root,
            output_root=output_root,
            checkpoint_root=checkpoint.parent if checkpoint else PROJECT_ROOT / "ckpt",
            artifact_root=artifact_root if artifact_root else PROJECT_ROOT / "outputs",
            cache_root=PROJECT_ROOT / ".cache",
            external_asset_root=PROJECT_ROOT / "third_party",
        )
    )
