"""Enumerate importable repository sources without hiding symlinked packages."""

import os
from pathlib import Path


def python_source_assets(repository: Path, *, prefix: str) -> dict[str, Path]:
    """Fail closed for links inside repositories (including dangling/cyclic links).

    Python imports follow directory links that pathlib's recursive glob omits.
    The repository root itself may be a resolved storage alias; entries within
    it must be ordinary files/directories so the inventory covers the sources.
    """
    if not repository.is_dir():
        raise FileNotFoundError(f"Model repository not found: {repository}")
    paths: list[Path] = []
    def failed(error: OSError) -> None:
        raise error
    for root, directories, files in os.walk(repository, followlinks=False, onerror=failed):
        for name in directories + files:
            path = Path(root) / name
            if path.is_symlink():
                raise ValueError(f"Cannot verify model implementation with a symlink inside its repository: {path}")
        paths.extend(Path(root) / name for name in files if name.endswith(".py"))
    return {f"{prefix}/{path.relative_to(repository).as_posix()}": path for path in sorted(paths)}
