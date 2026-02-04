"""Utilities for resolving scene paths and split files."""

from __future__ import annotations

from pathlib import Path


def resolve_scenes_base(scene_dir: Path) -> Path:
    """Resolve the base directory for scene NPZ files."""
    scenes_subdir = scene_dir / "scenes"
    return scenes_subdir if scenes_subdir.exists() else scene_dir


def load_split(scene_dir: Path, split_name: str) -> list[Path]:
    """Load a named split (train/val/test) from scene_dir."""
    split_path = scene_dir / f"{split_name}.txt"
    if not split_path.exists():
        return sorted(resolve_scenes_base(scene_dir).glob("*.npz"))
    base = resolve_scenes_base(scene_dir)
    paths: list[Path] = []
    for line in split_path.read_text().splitlines():
        name = line.strip()
        if name:
            paths.append(base / name)
    return paths


def load_split_file(scene_dir: Path, split_file: str | Path) -> list[Path]:
    """Load scene list from a split file path."""
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = scene_dir / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    base = resolve_scenes_base(scene_dir)
    paths: list[Path] = []
    for line in split_path.read_text().splitlines():
        name = line.strip()
        if name:
            paths.append(base / name)
    return paths


def resolve_scene_files(
    scene_dir: Path,
    *,
    split: str | None = None,
    split_file: str | Path | None = None,
) -> list[Path]:
    """Resolve scene NPZ files from split name or split file."""
    if split and split_file:
        raise ValueError("Provide only one of split or split_file.")
    if split_file is not None:
        return load_split_file(scene_dir, split_file)
    if split is not None:
        return load_split(scene_dir, split)
    return sorted(resolve_scenes_base(scene_dir).glob("*.npz"))
