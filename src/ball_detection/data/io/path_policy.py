"""Filesystem path and write policy guards."""

from __future__ import annotations

from pathlib import Path

from src.ball_detection.data.type import PathPolicy


def resolve_under_root(path: Path, *, policy: PathPolicy) -> Path:
    """Resolve a path and ensure it remains under policy.root_dir."""
    root = policy.root_dir.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes root_dir: {resolved} (root={root})") from exc
    return resolved


def ensure_writable(path: Path, *, policy: PathPolicy) -> None:
    """Validate overwrite rules before writing to a path."""
    target = resolve_under_root(path, policy=policy)
    if target.exists() and not policy.allow_overwrite:
        raise FileExistsError(f"Refusing overwrite (allow_overwrite=false): {target}")
