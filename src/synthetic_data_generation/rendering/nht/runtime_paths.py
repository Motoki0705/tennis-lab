"""Locate package paths inside the configured NHT runtime."""

from __future__ import annotations

from pathlib import Path


def package_root_from_file(package_file: str | Path) -> Path:
    """Return the checkout root containing an editable package module."""
    module_file = Path(package_file).resolve()
    package_root = module_file.parent.parent
    if not module_file.is_file():
        raise FileNotFoundError(
            f"Installed package module does not exist: {module_file}"
        )
    return package_root


def installed_gsplat_root() -> Path:
    """Return the gsplat package root imported by the active NHT Python."""
    import gsplat

    if gsplat.__file__ is None:
        raise RuntimeError("The active gsplat package has no filesystem origin.")
    return package_root_from_file(gsplat.__file__)
