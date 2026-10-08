"""Pose-independent window selection and exact shared-evaluation identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.checksum import dual_sha256

from .annotation_states import annotation_states
from .play_intervals import PlayIntervalConfig
from .play_manifest import load_pose_snapshot
from .store import BallFrameStore, ClipRecord, shard_name
from .temporal_sampling import TemporalSamplingConfig, select_sampled_windows

COORDINATE_MANIFEST_SCHEMA = "ball_coordinate_windows.v1"


def clip_semantic_digest(store: BallFrameStore, clip: ClipRecord) -> str:
    """Ignore store-global offsets; retain every input/GT/time/split contract."""
    digest = hashlib.sha256()
    descriptor = asdict(clip)
    descriptor.pop("index")
    descriptor["track_ids"] = sorted(clip.track_ids)
    digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
    rows = store.clip_rows(clip)
    for name in ("frame_index", "pts", "annotated", "is_target", "segment_break", "event", "inst_count"):
        digest.update(name.encode())
        digest.update(store.frames[name][rows].tobytes())
    first = int(store.frames["inst_start"][rows[0]])
    last = first + int(store.frames["inst_count"][rows].sum())
    tracks = [clip.track_ids[int(i)] for i in store.instances["track_index"][first:last]]
    digest.update(json.dumps(tracks, separators=(",", ":")).encode())
    for name in ("point_kind", "occluded", "xy"):
        values = store.instances[name][first:last].copy()
        if name == "xy":
            values[np.isnan(values)] = np.nan
            values[values == 0] = 0  # +0 and -0 denote the same position.
        digest.update(name.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def shared_pose_reference(
    store: BallFrameStore, pose_directory: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Freeze approved membership and prove matching GT; never join on ID alone."""
    manifest, reference_store, pose_digest = load_pose_snapshot(pose_directory)
    available = {clip.clip_id: clip for clip in store.clips}
    approved: dict[str, dict[str, Any]] = {}
    clip_hashes: dict[str, str] = {}
    for entry in manifest["clips"]:
        if entry["pose_status"] != "approved":
            continue
        clip_id = entry["clip_id"]
        if clip_id not in available:
            raise ValueError(f"Approved pose clip is absent from selected ball store: {clip_id}")
        reference_clip = reference_store.clip_by_id(clip_id)
        actual = clip_semantic_digest(store, available[clip_id])
        if actual != clip_semantic_digest(reference_store, reference_clip):
            raise ValueError(f"Shared evaluation GT/split/timeline mismatch: {clip_id}")
        approved[clip_id] = dict(entry)
        clip_hashes[clip_id] = actual
    if not approved:
        raise ValueError("Common evaluation requires approved pose clips")
    reference = dict(
        pose_manifest_sha256=pose_digest, ball_store=manifest["ball_store"],
        clip_ids=sorted(approved), clip_semantic_sha256=clip_hashes,
        matching="clip ID, source/group/camera, split, source/stored size, frame/PTS, GT and annotation/media identity",
    )
    return reference, approved


def build_coordinate_manifest(
    store: BallFrameStore, play_config: PlayIntervalConfig, sampling: TemporalSamplingConfig, *,
    input_kind: str, pose_directory: Path | None = None,
    common_reference: dict[str, Any] | None = None,
    approved_poses: dict[str, dict[str, Any]] | None = None,
    shard_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Prepare windows using ball labels; the pose-generation presence gate is absent.

    When supplied by the preparation transaction, shard hashes have already
    been computed on its frozen store. Standalone callers hash each used shard.
    """
    if input_kind not in {"mdd_only", "mdd_pose"}:
        raise ValueError("Unknown coordinate input kind")
    requires_pose = input_kind == "mdd_pose"
    if requires_pose and (pose_directory is None or approved_poses is None):
        raise ValueError("Pose-conditioned windows require a frozen approved pose membership")
    if not requires_pose and pose_directory is not None:
        raise ValueError("MDD-only manifest must not have a pose file dependency")
    identity = {name: dual_sha256(store.directory / name) for name in ("metadata.json", "index.npz")}
    store = BallFrameStore(store.directory)
    common_ids = set(common_reference["clip_ids"]) if common_reference is not None else set()
    known = {clip.clip_id for clip in store.clips}
    if common_ids - known:
        raise ValueError("Common evaluation contains clips outside the selected ball snapshot")
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    counts: dict[str, dict[str, Any]] = {}
    for clip in store.clips:
        if requires_pose and (approved_poses is None or clip.clip_id not in approved_poses):
            skipped.append(dict(clip_id=clip.clip_id, reason="pose_not_approved"))
            continue
        states = annotation_states(store, clip)
        play, windows = select_sampled_windows(states, play_config, sampling)
        by_step = {str(step): sum(w.frame_step == step for w in windows) for step in sampling.frame_steps}
        if not windows:
            skipped.append(dict(clip_id=clip.clip_id, reason="no_eligible_window"))
            continue
        shard = store.directory / "shards" / shard_name(clip.index)
        digest = shard_hashes[clip.clip_id] if shard_hashes is not None else dual_sha256(shard)
        semantic = clip_semantic_digest(store, clip)
        if clip.clip_id in common_ids:
            assert common_reference is not None
            if semantic != common_reference["clip_semantic_sha256"][clip.clip_id]:
                raise ValueError(f"Common evaluation identity changed: {clip.clip_id}")
        record: dict[str, Any] = dict(
            clip=asdict(clip), semantic_sha256=semantic,
            rgb_shard_sha256=digest, play=play,
            windows=[asdict(window) for window in windows], windows_by_step=by_step,
            common_evaluation=clip.clip_id in common_ids,
        )
        if requires_pose:
            assert approved_poses is not None
            record["pose"] = dict(approved_poses[clip.clip_id])
        records.append(record)
        group = counts.setdefault(f"{clip.source}/{clip.split}", dict(
            clips=0, frames=0, common_clips=0, windows_by_step={str(step): 0 for step in sampling.frame_steps},
        ))
        group["clips"] += 1
        group["frames"] += clip.frame_count
        group["common_clips"] += int(clip.clip_id in common_ids)
        for step, count in by_step.items():
            group["windows_by_step"][step] += count
    if not records:
        raise ValueError("No eligible coordinate training windows")
    if any(dual_sha256(store.directory / name) != expected for name, expected in identity.items()):
        raise ValueError("Ball snapshot changed while preparing coordinate windows")
    return dict(
        schema=COORDINATE_MANIFEST_SCHEMA, input_kind=input_kind,
        ball_store=dict(directory=str(store.directory), hashes=identity),
        pose_directory=str(pose_directory) if requires_pose else None,
        config=asdict(play_config), sampling=asdict(sampling),
        semantics=dict(mdd="recompute_after_rgb_subsampling", timestamps="real_seconds",
                       start_stride="sampled_frames", window_length=32,
                       clip_presence_gate=None, train_fps_mixture="uniform_frame_step"),
        common_evaluation=common_reference, clips=records, skipped=skipped, counts=counts,
    )
