"""Validate artifact identity, image correspondence and observation masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tennis_scene.chat_annotation.player_pose.storage import digest


def within(path: Path, *roots: Path) -> Path:
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f"Player artifact resolves outside configured roots: {path}")
    return resolved


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def validate_join(public: BallFrameStore, source: BallFrameStore, clip_id: str) -> None:
    """Allow appended/reindexed stores, but never different pixels or time axes."""
    current, original = public.clip_by_id(clip_id), source.clip_by_id(clip_id)
    for key in ("width", "height", "frame_count", "time_base", "fps", "media_sha256"):
        if getattr(current, key) != getattr(original, key):
            raise ValueError(f"Player/image clip identity differs: {key}")
    a, b = public.clip_rows(current), source.clip_rows(original)
    for key in ("frame_index", "pts", "offset", "length"):
        if not np.array_equal(public.frames[key][a], source.frames[key][b]):
            raise ValueError(f"Player/image frame correspondence differs: {key}")
    paths = [
        within(store.directory / SHARDS_DIR / shard_name(clip.index), store.directory)
        for store, clip in ((public, current), (source, original))
    ]
    # Migrated shards are hard links. Independently copied stores need one SHA
    # comparison per cached clip, never image decoding on the playback path.
    if file_stamp(paths[0])[:2] != file_stamp(paths[1])[:2] and digest(
        paths[0]
    ) != digest(paths[1]):
        raise ValueError("Player poses belong to different JPEG images")


def read_raw(root: Path, store: BallFrameStore, clip_id: str) -> dict[str, Any]:
    record = read_object(within(root / "generation.json", root))
    clip = store.clip_by_id(clip_id)
    if (
        record.get("status") != "complete"
        or record.get("clip_id") != clip_id
        or record.get("frame_count") != clip.frame_count
    ):
        raise ValueError(
            "Raw generation identity/status differs from the selected clip"
        )
    input_path = within(root / "input.json", root)
    if digest(input_path) != record.get("files", {}).get("input.json"):
        raise ValueError("Raw input checksum mismatch")
    source = read_object(input_path)
    shard = within(
        store.directory / SHARDS_DIR / shard_name(clip.index), store.directory
    )
    if source.get("clip_id") != clip_id or source.get("shard_sha256") != digest(shard):
        raise ValueError("Raw tracking was generated from different JPEG images")
    path = within(root / "tracks.npz", root)
    if digest(path) != record.get("files", {}).get("tracks.npz"):
        raise ValueError("Raw tracks checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "track_ids",
        "frame_index",
        "pts",
        "boxes",
        "keypoints",
        "detection_rows",
    }
    if not required <= data.keys():
        raise ValueError("Raw tracks are missing required arrays")
    ids = data["track_ids"]
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Raw track IDs must be unique integers")
    frames, people = clip.frame_count, len(ids)
    if record.get("raw_tracks") != people:
        raise ValueError("Raw track count differs from generation record")
    for name, shape in {
        "boxes": (people, frames, 4),
        "keypoints": (people, frames, 17, 3),
        "detection_rows": (people, frames),
    }.items():
        if data[name].shape != shape:
            raise ValueError(f"Raw pose axis mismatch: {name}")
    for name in ("boxes", "keypoints"):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite raw pose coordinates: {name}")
    rows = store.clip_rows(clip)
    for name in ("frame_index", "pts"):
        if data[name].dtype.kind not in "iu" or not np.array_equal(
            data[name], store.frames[name][rows]
        ):
            raise ValueError(f"Raw pose/frame time axes differ: {name}")
    detection_rows = data["detection_rows"]
    if detection_rows.dtype.kind not in "iu" or (detection_rows < -1).any():
        raise ValueError("Invalid raw observation mask")
    # Interpolated tracker boxes may exist with detection_rows=-1. They are NOT
    # pose observations and never become visible through this adapter.
    return {
        "player_ids": np.asarray([f"raw_{i}" for i in ids]),
        "boxes_xyxy": data["boxes"].transpose(1, 0, 2),
        "keypoints": data["keypoints"].transpose(1, 0, 2, 3),
        "observed": (detection_rows >= 0).T,
        "raw_track_ids": np.broadcast_to(ids, (frames, people)),
        "detection_rows": detection_rows.T,
    }


def frame_people(
    data: dict[str, Any], frame: int, breaks: Any, *, trail_length: int = 20
) -> list[dict[str, Any]]:
    people = []
    for person in np.flatnonzero(data["observed"][frame]):
        box = data["boxes_xyxy"][frame, person]
        trail = []
        for previous in range(frame, max(-1, frame - trail_length), -1):
            if not data["observed"][previous, person]:
                break
            xyxy = data["boxes_xyxy"][previous, person]
            trail.append([float((xyxy[0] + xyxy[2]) / 2), float(xyxy[3])])
            if breaks[previous]:
                break
        people.append(
            {
                "id": str(data["player_ids"][person]),
                "raw_track_id": int(data["raw_track_ids"][frame, person]),
                "box": box.tolist(),
                "keypoints": data["keypoints"][frame, person].tolist(),
                "trail": trail[::-1],
            }
        )
    return people
