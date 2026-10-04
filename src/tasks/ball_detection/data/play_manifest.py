"""Freeze approved pose clips and annotation-derived play-window proposals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.play_intervals import (
    PlayIntervalConfig,
    infer_play_intervals,
    mask_intervals,
)
from src.tasks.ball_detection.data.store import (
    POINT_KIND_CODES,
    BallFrameStore,
    ClipRecord,
    shard_name,
)
from src.utils.checksum import dual_sha256


def clip_evidence(
    store: BallFrameStore, clip: ClipRecord,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    rows = store.clip_rows(clip)
    eligible = store.frames["is_target"][rows].copy()
    reviewed = store.frames["annotated"][rows] & eligible
    single = store.frames["inst_count"][rows] == 1
    kind = np.full(clip.frame_count, 255, np.uint8)
    offsets = store.frames["inst_start"][rows]
    kind[single] = store.instances["point_kind"][offsets[single]]
    presence = reviewed & single & (kind != POINT_KIND_CODES["out_of_frame"])
    observed = reviewed & single & (kind == POINT_KIND_CODES["observed"])
    pts = store.frames["pts"][rows]
    times = (pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))
    return presence, observed, eligible, times


def build_play_manifest(pose_directory: Path, config: PlayIntervalConfig) -> dict[str, Any]:
    """Snapshot once; never follow a growing pose manifest during experiments."""
    pose_path = pose_directory / "manifest.json"
    raw = pose_path.read_bytes()
    pose = json.loads(raw)
    if pose["schema"] != "ball_detection_player_poses.v1" or pose["coordinate_system"] != "stored_jpeg_pixels":
        raise ValueError("Unsupported pose dataset")
    store_root = Path(pose["ball_store"]["directory"])
    for name, digest in pose["ball_store"]["hashes"].items():
        if dual_sha256(store_root / name) != digest:
            raise ValueError("Pose snapshot and ball store identity differ")
    store = BallFrameStore(store_root)
    if {e["clip_id"] for e in pose["clips"]} != {c.clip_id for c in store.clips}:
        raise ValueError("Pose manifest does not cover its declared snapshot")
    counts: dict[str, dict[str, int]] = {}
    clips: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in pose["clips"]:
        if entry["pose_status"] != "approved":
            skipped.append({"clip_id": entry["clip_id"], "reason": entry["pose_status"]})
            continue
        clip = store.clip_by_id(entry["clip_id"])
        presence, observed, eligible, times = clip_evidence(store, clip)
        selection = infer_play_intervals(presence, observed, times, eligible, config)
        group = counts.setdefault(f"{clip.source}/{clip.split}", dict(
            clips=0, frames=0, play_frames=0, selected_frames=0, observed_frames=0,
            excluded_observed_frames=0, windows=0, bridged_frames=0,
        ))
        values = dict(clips=1, frames=clip.frame_count,
                      play_frames=int(selection.play.sum()), selected_frames=int(selection.selected.sum()), observed_frames=int(observed.sum()),
                      excluded_observed_frames=int((observed & ~selection.selected).sum()),
                      windows=len(selection.window_starts), bridged_frames=int(selection.bridged.sum()))
        for key, value in values.items():
            group[key] += value
        clips.append(dict(
            clip=asdict(clip), pose=dict(entry),
            rgb_shard_sha256=dual_sha256(store.directory / "shards" / shard_name(clip.index)),
            play=selection.intervals,
            excluded=selection.excluded, training_intervals=mask_intervals(selection.selected),
            windows=selection.window_starts, counts=values,
        ))
    if not clips:
        raise ValueError("No approved pose clips exist")
    # All pose file identities are copied from this single manifest snapshot.
    # Later approvals may proceed; reading them never grows this frozen subset.
    return dict(
        schema="ball_play_windows.v1", pose_directory=str(pose_directory.resolve()),
        pose_manifest_sha256=hashlib.sha256(raw).hexdigest(), ball_store=pose["ball_store"],
        config=asdict(config), clips=clips, skipped=skipped, counts=counts,
        semantics="Annotation-based selection proposal; excluded does not certify non-play. No label interpolation.",
    )
