"""Strict discovery of the canonical dataset/video clip-studio layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.utils.configuration import PathResolver, PathRole

_VIDEO_ID_PATTERN = re.compile(r"^video_(\d{3,})$")
_CAMERA_FILE_PATTERN = re.compile(r"^cam(\d+)\.mp4$")


@dataclass(frozen=True, slots=True)
class ClipStudioLayout:
    """Resolved paths belonging to one video in one tennis dataset."""

    dataset_id: str
    video_id: str
    source_directory: Path
    video_paths: tuple[Path, ...]
    camera_ids: tuple[str, ...]
    projects_path: Path
    dataset_directory: Path


def discover_clip_studio_layout(
    resolver: PathResolver,
    source_directory: str,
) -> ClipStudioLayout:
    """Discover one ``raw/<dataset>/<video>`` source without path fallback."""
    source = resolver.resolve(PathRole.DATA, source_directory)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")

    relative = source.relative_to(resolver.roots.data_root)
    parts = relative.parts
    if len(parts) != 4 or parts[:2] != ("tennis_multivew", "raw"):
        raise ValueError(
            "source_directory must be "
            "tennis_multivew/raw/<dataset_id>/<video_id> relative to paths.data_root; "
            f"got {source_directory!r}"
        )
    dataset_id, video_id = parts[2], parts[3]
    if not dataset_id or dataset_id in {".", ".."}:
        raise ValueError(
            f"invalid dataset_id derived from source_directory: {dataset_id!r}"
        )
    if _VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        raise ValueError(
            f"video_id must match 'video_' followed by at least three digits, got {video_id!r}"
        )

    mp4_files = sorted(path for path in source.iterdir() if path.suffix == ".mp4")
    unexpected = [
        path.name
        for path in mp4_files
        if _CAMERA_FILE_PATTERN.fullmatch(path.name) is None
    ]
    if unexpected:
        raise ValueError(
            f"source directory contains unsupported MP4 files: {unexpected}; "
            "camera files must be named cam<index>.mp4"
        )
    indexed = sorted(
        (int(match.group(1)), path)
        for path in mp4_files
        if (match := _CAMERA_FILE_PATTERN.fullmatch(path.name)) is not None
    )
    if not indexed:
        raise ValueError(f"source directory contains no cam<index>.mp4 files: {source}")
    indices = [index for index, _ in indexed]
    expected = list(range(len(indexed)))
    if indices != expected:
        raise ValueError(
            f"camera indices must be contiguous from cam0, got {indices} in {source}"
        )

    processed = resolver.resolve(
        PathRole.DATA,
        "tennis_multivew/processed",
        dataset_id,
    )
    return ClipStudioLayout(
        dataset_id=dataset_id,
        video_id=video_id,
        source_directory=source,
        video_paths=tuple(path for _, path in indexed),
        camera_ids=tuple(f"cam{index}" for index in indices),
        projects_path=resolver.resolve_beneath(
            PathRole.DATA, processed, "projects.json"
        ),
        dataset_directory=resolver.resolve_beneath(PathRole.DATA, processed, "dataset"),
    )


__all__ = ["ClipStudioLayout", "discover_clip_studio_layout"]
