"""Enumerate importable repository sources without hiding symlinked packages."""

import os
import sys
from collections.abc import Mapping
from importlib.machinery import NamespaceLoader
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

    Ordinary modules require a declared Python file. Genuine namespace packages
    execute no initializer; every search directory must be a parent of declared
    sources. Sourceless/extension modules cannot stand in for Python sources.
    """
    declared = {path.resolve() for path in assets.values()}
    root = repository.resolve()
    directories = {parent for path in declared for parent in path.parents if parent == root or parent.is_relative_to(root)}
    for name, module in tuple(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        source = getattr(module, '__file__', None)
        if source is None and isinstance(getattr(module, '__loader__', None), NamespaceLoader):
            search = tuple(getattr(module, '__path__', ()))
            if search and all(isinstance(path, str) and Path(path).is_absolute() and Path(path).resolve() in directories for path in search):
                continue
        if not isinstance(source, str) or not Path(source).resolve().is_relative_to(root) or Path(source).resolve() not in declared:
            raise RuntimeError(f'Module {name} resolved outside the declared repository source inventory: {source!r}; use a fresh process')
