"""Source selection and identity shared by detector and context caches."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.store import (
    SOURCES,
    SPLIT_CODES,
    BallFrameStore,
    ClipRecord,
)
from src.utils.checksum import dual_sha256


def clip_record(clip: ClipRecord) -> dict[str, Any]:
    return {**asdict(clip), "track_ids": list(clip.track_ids)}


def store_hashes(directory: Path) -> dict[str, str]:
    return {name: dual_sha256(directory / name) for name in ("metadata.json", "index.npz")}


def select_clips(
    store: BallFrameStore, *, splits: tuple[str, ...], sources: tuple[str, ...],
) -> tuple[ClipRecord, ...]:
    """Select only explicit source/split pairs; never select by label quality."""
    for name, requested, allowed in (("splits", splits, SPLIT_CODES), ("sources", sources, SOURCES)):
        if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(allowed):
            raise ValueError(f"Invalid or repeated {name}: {requested}")
    groups: dict[tuple[str, str], str] = {}
    for clip in store.clips:
        group = (clip.source, clip.group_id)
        if group in groups and groups[group] != clip.split:
            raise ValueError(f"Source group crosses splits: {group}")
        groups[group] = clip.split
    clips = tuple(clip for clip in store.clips if clip.split in splits and clip.source in sources)
    if {(clip.source, clip.split) for clip in clips} != {(source, split) for source in sources for split in splits}:
        raise ValueError("Every requested source/split pair must contain at least one clip")
    return clips


