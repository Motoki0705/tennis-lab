"""Enumerate importable repository sources without hiding symlinked packages."""

import os
import sys
from collections.abc import Mapping
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


def validate_source_modules(repository: Path, assets: Mapping[str, Path], prefixes: tuple[str, ...]) -> None:
    """Reject absolute-import/sys.path escapes, including package submodules.

    Membership in the declared file inventory is stronger than containment:
    namespace packages and sourceless/extension modules cannot silently stand
    in for one of the declared Python files.
    """
    declared = {path.resolve() for path in assets.values()}
    root = repository.resolve()
    for name, module in tuple(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        source = getattr(module, '__file__', None)
        if not isinstance(source, str) or not Path(source).resolve().is_relative_to(root) or Path(source).resolve() not in declared:
            raise RuntimeError(f'Module {name} resolved outside the declared repository source inventory: {source!r}; use a fresh process')
