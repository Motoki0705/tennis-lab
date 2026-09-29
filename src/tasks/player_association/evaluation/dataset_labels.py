"""Dataset-owned evaluation labels; no repository fixtures or legacy fallback."""

from pathlib import Path

from src.tasks.player_association.evaluation.labels import ClipLabels


def label_path(dataset: Path, clip_id: str) -> Path:
    parts = clip_id.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") or "\\" in part for part in parts):
        raise ValueError(f"Expected video/clip ID, got {clip_id!r}")
    video, clip = parts
    return dataset / "videos" / video / "clips" / clip / "annotations/player_association/labels.json"


def discover_labels(dataset: Path) -> tuple[Path, ...]:
    """Reject empty inputs and misplaced clip IDs before evaluating/calibrating."""
    paths = tuple(sorted(dataset.glob("videos/*/clips/*/annotations/player_association/labels.json")))
    if not paths:
        raise ValueError(f"No player association labels under {dataset}")
    for path in paths:
        labels = ClipLabels.load(path)
        if label_path(dataset, labels.clip_id) != path:
            raise ValueError(f"Label clip ID {labels.clip_id!r} disagrees with {path}")
    return paths
