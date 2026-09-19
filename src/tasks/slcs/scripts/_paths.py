"""Path roots shared by the explicit SLCS analysis command boundaries."""

from os.path import commonpath
from pathlib import Path

from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def cli_resolver(output_root: Path, inputs: tuple[Path, ...] = ()) -> PathResolver:
    """Keep output authority explicit and accept caller-selected absolute inputs."""
    if not output_root.is_absolute() or any(not path.is_absolute() for path in inputs):
        raise ValueError("CLI roots and input paths must be absolute")
    input_root = (
        Path(
            commonpath(
                [str(output_root.resolve()), *(str(path.resolve()) for path in inputs)]
            )
        )
        if inputs
        else PROJECT_ROOT
    )
    return PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT.resolve(),
            data_root=(PROJECT_ROOT / "data").resolve(),
            checkpoint_root=(PROJECT_ROOT / "ckpt").resolve(),
            artifact_root=input_root.resolve(),
            output_root=output_root.resolve(),
            cache_root=(PROJECT_ROOT / ".cache").resolve(),
            external_asset_root=(PROJECT_ROOT / "third_party").resolve(),
        )
    )
