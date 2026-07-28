"""Locate Git provenance from packages installed in the isolated NHT runtime."""

from __future__ import annotations

from pathlib import Path


def repository_from_package_file(package_file: str | Path) -> Path:
    """Return the Git repository containing an editable package module."""
    module_file = Path(package_file).resolve()
    repository = module_file.parent.parent
    if not module_file.is_file():
        raise FileNotFoundError(
            f"Installed package module does not exist: {module_file}"
        )
    if not (repository / ".git").exists():
        raise RuntimeError(
            f"Installed package is not rooted in a Git checkout: {module_file}"
        )
    return repository


def installed_gsplat_repository() -> Path:
    """Return the exact gsplat checkout imported by the active NHT Python."""
    import gsplat

    if gsplat.__file__ is None:
        raise RuntimeError("The active gsplat package has no filesystem origin.")
    return repository_from_package_file(gsplat.__file__)
