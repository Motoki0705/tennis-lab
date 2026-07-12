"""Path conventions for clip studio source recordings and processed data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/tennis_multivew")
DEFAULT_CAMERA_IDS = ("cam0", "cam1", "cam2")


@dataclass(frozen=True)
class StandardClipStudioPaths:
    """Standard paths belonging to one multi-camera match."""

    video_paths: tuple[Path, ...]
    project_path: Path
    dataset_dir: Path


def standard_clip_studio_paths(
    data_root: str | Path, match_id: str
) -> StandardClipStudioPaths:
    """Build the standard raw and processed paths for one match."""
    if not match_id or Path(match_id).name != match_id or match_id in {".", ".."}:
        raise ValueError(f"match_id must be a single path component, got {match_id!r}")

    root = Path(data_root)
    raw_match_dir = root / "raw" / match_id
    processed_match_dir = root / "processed" / match_id
    return StandardClipStudioPaths(
        video_paths=tuple(
            raw_match_dir / f"{camera_id}.mp4" for camera_id in DEFAULT_CAMERA_IDS
        ),
        project_path=processed_match_dir / "project.json",
        dataset_dir=processed_match_dir / "dataset",
    )


__all__ = [
    "DEFAULT_CAMERA_IDS",
    "DEFAULT_DATA_ROOT",
    "StandardClipStudioPaths",
    "standard_clip_studio_paths",
]
