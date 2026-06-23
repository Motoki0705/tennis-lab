"""Project path resolution helpers.

Consolidates the ``Path(__file__).resolve().parents[N]`` / Hydra
``to_absolute_path`` idioms that were re-implemented across tasks. Resolve
relative paths through :func:`resolve_project_path` (or join onto
:data:`PROJECT_ROOT`) instead of recomputing the repository root with a
hand-counted ``parents[N]`` depth at every call site.
"""

from __future__ import annotations

from pathlib import Path

# ``src/utils/paths.py`` -> parents[0] = src/utils, [1] = src, [2] = repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root.

    ``~`` is expanded first. Absolute paths are returned resolved as-is;
    relative paths are joined onto :data:`PROJECT_ROOT`.

    Args:
        path: Absolute or repo-relative path.

    Returns:
        The resolved absolute path.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


__all__ = ["PROJECT_ROOT", "resolve_project_path"]
