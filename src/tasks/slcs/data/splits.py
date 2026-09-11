"""Video-level train/val/test splits for the issue #634 dataset.

Leak prevention: the split unit is the ``video_id`` — every clip cut from
the same multi-camera video lands in the same split. Assignment is deterministic
given ``(seed, ratios, videos)``
via :func:`src.utils.data.splits.make_group_split_map`, weighted by total clip
frames so long videos do not distort the ratios.

The split file is JSON::

    {
      "format_version": 1,
      "seed": 0,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "assignments": {"<video_id>": "train" | "val" | "test", ...}
    }

Loading is strict: a video present in the dataset but missing from the
split file (or vice versa) is an error, so stale split files fail loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetManifestError,
)
from src.utils.data.splits import GroupSplitConfig, make_group_split_map
from src.utils.io import load_json, save_json_atomic

SPLIT_FORMAT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


def generate_video_splits(
    index: SLCSDataIndex,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign every video_id in the index to a split deterministically."""
    weights: dict[str, int] = {}
    for ref in index.clips:
        manifest = ClipManifest.load(index.clip_dir(ref))
        weights[ref.video_id] = weights.get(ref.video_id, 0) + manifest.num_frames
    if not weights:
        raise DatasetManifestError(
            f"dataset at {index.root} contains no clips to split."
        )
    assignments: dict[str, str] = make_group_split_map(
        weights,
        GroupSplitConfig(val_ratio=val_ratio, test_ratio=test_ratio, seed=seed),
    )
    return assignments


def generate_overfit_splits(index: SLCSDataIndex) -> dict[str, str]:
    """Assign every video to train for an explicit memorization experiment."""
    video_ids = index.video_ids()
    if not video_ids:
        raise DatasetManifestError(f"dataset at {index.root} contains no videos.")
    return {video_id: "train" for video_id in video_ids}


def save_split_file(
    path: str | Path,
    assignments: dict[str, str],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Path:
    """Write a split file (atomic)."""
    for video_id, split in assignments.items():
        if split not in SPLIT_NAMES:
            raise DatasetManifestError(
                f"assignment {video_id!r} -> {split!r} is not one of {SPLIT_NAMES}."
            )
    payload: dict[str, Any] = {
        "format_version": SPLIT_FORMAT_VERSION,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "assignments": assignments,
    }
    return Path(save_json_atomic(payload, path))


def load_split_assignments(path: str | Path, index: SLCSDataIndex) -> dict[str, str]:
    """Load a split file and verify it exactly covers the dataset's videos."""
    split_path = Path(path)
    if not split_path.is_file():
        raise DatasetManifestError(f"split file not found: {split_path}")
    payload = load_json(split_path)
    if not isinstance(payload, dict):
        raise DatasetManifestError(f"{split_path} must contain a JSON object.")
    if payload.get("format_version") != SPLIT_FORMAT_VERSION:
        raise DatasetManifestError(
            f"{split_path} declares format_version={payload.get('format_version')!r}; "
            f"supported: {SPLIT_FORMAT_VERSION}."
        )
    assignments_raw = payload.get("assignments")
    if not isinstance(assignments_raw, dict) or not assignments_raw:
        raise DatasetManifestError(
            f"{split_path} must contain a non-empty 'assignments' map."
        )
    assignments = {str(k): str(v) for k, v in assignments_raw.items()}
    for video_id, split in assignments.items():
        if split not in SPLIT_NAMES:
            raise DatasetManifestError(
                f"{split_path}: {video_id!r} assigned to unknown split {split!r}."
            )

    dataset_videos = set(index.video_ids())
    split_videos = set(assignments)
    missing = dataset_videos - split_videos
    stale = split_videos - dataset_videos
    if missing or stale:
        raise DatasetManifestError(
            f"{split_path} does not match the dataset: "
            f"videos missing from the split file: {sorted(missing)}; "
            f"stale split entries with no dataset video: {sorted(stale)}. "
            "Regenerate the split file (scripts/make_splits.py)."
        )
    return assignments


__all__ = [
    "SPLIT_FORMAT_VERSION",
    "SPLIT_NAMES",
    "generate_video_splits",
    "generate_overfit_splits",
    "load_split_assignments",
    "save_split_file",
]
