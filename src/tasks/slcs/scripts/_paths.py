"""Path roots shared by the explicit SLCS analysis command boundaries."""

from pathlib import Path

from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def cli_resolver(output_root: Path) -> PathResolver:
    """Keep output authority independent of caller-selected input locations."""
    if not output_root.is_absolute():
        raise ValueError("CLI roots must be absolute")
    return PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT.resolve(),
            data_root=(PROJECT_ROOT / "data").resolve(),
            checkpoint_root=(PROJECT_ROOT / "ckpt").resolve(),
            artifact_root=PROJECT_ROOT.resolve(),
            output_root=output_root.resolve(),
            cache_root=(PROJECT_ROOT / ".cache").resolve(),
            external_asset_root=(PROJECT_ROOT / "third_party").resolve(),
        )
    )
