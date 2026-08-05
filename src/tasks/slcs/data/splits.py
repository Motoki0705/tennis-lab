"""Recording-level train/val/test splits for the issue #634 dataset.

Leak prevention: the split unit is the ``recording_id`` — every clip cut from
the same recording (and therefore from the same source videos) lands in the
same split. Assignment is deterministic given ``(seed, ratios, recordings)``
via :func:`src.utils.data.splits.make_group_split_map`, weighted by total clip
frames so long recordings do not distort the ratios.

The split file is JSON::

    {
      "format_version": 1,
      "seed": 0,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "assignments": {"<recording_id>": "train" | "val" | "test", ...}
    }

Loading is strict: a recording present in the dataset but missing from the
split file (or vice versa) is an error, so stale split files fail loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tasks.slcs.data.contract import (
    ClipManifest,
    DatasetContractError,
    DatasetIndex,
)
from src.utils.data.splits import GroupSplitConfig, make_group_split_map
from src.utils.io import load_json, save_json_atomic

SPLIT_FORMAT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


def generate_recording_splits(
    index: DatasetIndex,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign every recording_id in the index to a split deterministically."""
    weights: dict[str, int] = {}
    for ref in index.clips:
        manifest = ClipManifest.load(index.clip_dir(ref))
        weights[ref.recording_id] = (
            weights.get(ref.recording_id, 0) + manifest.num_frames
        )
    if not weights:
        raise DatasetContractError(
            f"dataset at {index.root} contains no clips to split."
        )
    assignments: dict[str, str] = make_group_split_map(
        weights,
        GroupSplitConfig(val_ratio=val_ratio, test_ratio=test_ratio, seed=seed),
    )
    return assignments


def generate_overfit_splits(index: DatasetIndex) -> dict[str, str]:
    """Assign every recording to train for an explicit memorization experiment."""
    recording_ids = index.recording_ids()
    if not recording_ids:
        raise DatasetContractError(f"dataset at {index.root} contains no recordings.")
    return {recording_id: "train" for recording_id in recording_ids}


def save_split_file(
    path: str | Path,
    assignments: dict[str, str],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Path:
    """Write a split file (atomic)."""
    for recording_id, split in assignments.items():
        if split not in SPLIT_NAMES:
            raise DatasetContractError(
                f"assignment {recording_id!r} -> {split!r} is not one of {SPLIT_NAMES}."
            )
    payload: dict[str, Any] = {
        "format_version": SPLIT_FORMAT_VERSION,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "assignments": assignments,
    }
    return Path(save_json_atomic(payload, path))


def load_split_assignments(path: str | Path, index: DatasetIndex) -> dict[str, str]:
    """Load a split file and verify it exactly covers the dataset's recordings."""
    split_path = Path(path)
    if not split_path.is_file():
        raise DatasetContractError(f"split file not found: {split_path}")
    payload = load_json(split_path)
    if not isinstance(payload, dict):
        raise DatasetContractError(f"{split_path} must contain a JSON object.")
    if payload.get("format_version") != SPLIT_FORMAT_VERSION:
        raise DatasetContractError(
            f"{split_path} declares format_version={payload.get('format_version')!r}; "
            f"supported: {SPLIT_FORMAT_VERSION}."
        )
    assignments_raw = payload.get("assignments")
    if not isinstance(assignments_raw, dict) or not assignments_raw:
        raise DatasetContractError(
            f"{split_path} must contain a non-empty 'assignments' map."
        )
    assignments = {str(k): str(v) for k, v in assignments_raw.items()}
    for recording_id, split in assignments.items():
        if split not in SPLIT_NAMES:
            raise DatasetContractError(
                f"{split_path}: {recording_id!r} assigned to unknown split {split!r}."
            )

    dataset_recordings = set(index.recording_ids())
    split_recordings = set(assignments)
    missing = dataset_recordings - split_recordings
    stale = split_recordings - dataset_recordings
    if missing or stale:
        raise DatasetContractError(
            f"{split_path} does not match the dataset: "
            f"recordings missing from the split file: {sorted(missing)}; "
            f"stale split entries with no dataset recording: {sorted(stale)}. "
            "Regenerate the split file (scripts/make_splits.py)."
        )
    return assignments


__all__ = [
    "SPLIT_FORMAT_VERSION",
    "SPLIT_NAMES",
    "generate_recording_splits",
    "generate_overfit_splits",
    "load_split_assignments",
    "save_split_file",
]
