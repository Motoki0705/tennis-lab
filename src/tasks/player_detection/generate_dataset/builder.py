"""Build ``data/player_detection/<version>`` from chat-annotation player labels.

Only frames whose annotation lists at least one player are encoded. Each clip
is written to its own JPEG shard by a worker process; the index and metadata
are assembled afterwards and the whole version directory is published with a
single atomic rename, so a partially built store is never visible.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.player_detection.configuration import GenerateDatasetConfig
from src.tasks.player_detection.data.store import (
    BBOX_SOURCE_CODES,
    FRAME_FIELDS,
    INDEX_FILE,
    INSTANCE_FIELDS,
    METADATA_FILE,
    SCHEMA_VERSION,
    SHARDS_DIR,
    SPLIT_CODES,
    PlayerFrameStore,
    shard_name,
)
from src.tasks.player_detection.generate_dataset.sources import (
    ClipSource,
    collect_clip_sources,
)
from src.tennis_scene.chat_annotation.runtime.media import check_clip, decode_range
from src.utils.data.splits import GroupSplitConfig, make_group_split_map


@dataclass(frozen=True, slots=True)
class ClipTables:
    """Frame/instance rows of one clip; offsets are relative to its shard."""

    frames: dict[str, NDArray[np.generic]]
    instances: dict[str, NDArray[np.generic]]
    track_ids: tuple[str, ...]


def clip_track_ids(source: ClipSource) -> tuple[str, ...]:
    """Deterministic clip-local track table (sorted annotation IDs)."""
    return tuple(
        sorted({player.track_id for frame in source.annotation.frames for player in frame.players})
    )


def encode_clip(source: ClipSource, shard_path: Path, jpeg_quality: int) -> ClipTables:
    """Decode the clip in presentation order and store its player frames."""
    manifest = source.manifest
    timeline = check_clip(source.video_path, manifest)
    track_ids = clip_track_ids(source)
    track_lookup = {track_id: index for index, track_id in enumerate(track_ids)}
    frames: dict[str, list[object]] = {name: [] for name in FRAME_FIELDS}
    instances: dict[str, list[object]] = {name: [] for name in INSTANCE_FIELDS}
    boxes: list[list[float]] = []
    offset = 0
    decoded = 0
    with shard_path.open("xb") as shard:
        for index, frame in enumerate(decode_range(source.video_path, timeline, 0, len(manifest.frames))):
            decoded += 1
            annotated = source.annotation.frames[index]
            if annotated.frame_index != index:
                raise ValueError(f"{source.clip_id}: annotation frame order mismatch at {index}")
            if not annotated.players:
                continue
            pixels = frame.to_ndarray(format="bgr24")
            if pixels.shape != (manifest.height, manifest.width, 3):
                raise ValueError(f"{source.clip_id}: decoded frame {index} has shape {pixels.shape}")
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise RuntimeError(f"{source.clip_id}: JPEG encoding failed at frame {index}")
            payload = encoded.tobytes()
            shard.write(payload)
            mapped = manifest.frames[index]
            for name, value in (
                ("frame_index", index),
                ("source_frame_index", mapped.source_frame_index),
                ("clip_pts", mapped.clip_pts),
                ("is_target", mapped.is_target),
                ("reviewed", annotated.reviewed),
                ("offset", offset),
                ("length", len(payload)),
                ("inst_count", len(annotated.players)),
            ):
                frames[name].append(value)
            offset += len(payload)
            for player in annotated.players:
                instances["track_index"].append(track_lookup[player.track_id])
                instances["bbox_source"].append(BBOX_SOURCE_CODES[player.bbox_source])
                instances["occluded"].append(player.occluded)
                instances["truncated"].append(player.truncated)
                boxes.append(
                    [np.nan] * 4
                    if player.bbox_xyxy is None
                    else [float(value) for value in player.bbox_xyxy]
                )
    if decoded != len(manifest.frames):
        raise ValueError(f"{source.clip_id}: decoded {decoded} of {len(manifest.frames)} frames")
    if not frames["frame_index"]:
        raise ValueError(f"{source.clip_id}: annotation contains no frame with players")
    frame_count = len(frames["frame_index"])
    frame_arrays: dict[str, NDArray[np.generic]] = {
        name: np.asarray(frames[name], dtype=FRAME_FIELDS[name])
        for name in FRAME_FIELDS
        if name not in {"clip", "inst_start"}
    }
    frame_arrays["clip"] = np.zeros(frame_count, dtype=np.int32)
    frame_arrays["inst_start"] = np.zeros(frame_count, dtype=np.int64)
    instance_arrays: dict[str, NDArray[np.generic]] = {
        name: np.asarray(instances[name], dtype=INSTANCE_FIELDS[name]) for name in INSTANCE_FIELDS
    }
    instance_arrays["bbox_xyxy"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return ClipTables(frame_arrays, instance_arrays, track_ids)


def _encode_job(job: tuple[ClipSource, Path, int]) -> ClipTables:
    source, shard_path, jpeg_quality = job
    return encode_clip(source, shard_path, jpeg_quality)


def assign_splits(sources: list[ClipSource], config: GenerateDatasetConfig) -> dict[str, str]:
    """Split by source video so no broadcast leaks across train/val/test."""
    weights: Counter[str] = Counter()
    for source in sources:
        weights[source.manifest.source.source_id] += sum(
            1 for frame in source.annotation.frames if frame.players
        )
    splits: dict[str, str] = make_group_split_map(
        dict(sorted(weights.items())),
        GroupSplitConfig(config.val_ratio, config.test_ratio, config.split_seed),
    )
    return splits


def merge_tables(tables: list[ClipTables]) -> dict[str, NDArray[np.generic]]:
    """Concatenate clip tables, setting clip indices and global instance starts."""
    columns: dict[str, list[NDArray[np.generic]]] = {
        name: [] for name in (*FRAME_FIELDS, *INSTANCE_FIELDS, "bbox_xyxy")
    }
    for clip_index, table in enumerate(tables):
        frames = dict(table.frames)
        frames["clip"] = np.full(len(frames["frame_index"]), clip_index, dtype=np.int32)
        for name in FRAME_FIELDS:
            if name != "inst_start":
                columns[name].append(frames[name])
        for name, values in table.instances.items():
            columns[name].append(values)
    merged = {name: np.concatenate(values) for name, values in columns.items() if name != "inst_start"}
    counts = merged["inst_count"].astype(np.int64)
    merged["inst_start"] = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return merged


def _summary_readme(metadata: dict[str, object]) -> str:
    counts = metadata["counts"]
    return (
        f"# player_detection dataset `{metadata['version']}`\n\n"
        "Generated by `python -m src.tasks.player_detection.scripts.generate_dataset`.\n"
        "Schema and access API: `src/tasks/player_detection/data/store.py`.\n\n"
        f"```json\n{json.dumps(counts, indent=2)}\n```\n"
    )


def build_dataset(config: GenerateDatasetConfig) -> Path:
    """Build and atomically publish one dataset version; returns its directory."""
    destination = config.dataset_dir
    if destination.exists():
        raise FileExistsError(
            f"Dataset version already exists: {destination}. Choose a new dataset.version."
        )
    collected = collect_clip_sources(config.annotation_root, allowed_statuses=config.allowed_statuses)
    # Only frames listing players are stored, so a clip without any player
    # contributes nothing; it is recorded in the metadata, not encoded.
    sources = [s for s in collected if any(frame.players for frame in s.annotation.frames)]
    kept = {s.clip_id for s in sources}
    without_players = sorted(s.clip_id for s in collected if s.clip_id not in kept)
    if not sources:
        raise ValueError("No processed player annotation contains a player")
    splits = assign_splits(sources, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".building-{destination.name}-{uuid.uuid4().hex[:8]}"
    (staging / SHARDS_DIR).mkdir(parents=True)
    try:
        jobs = [
            (source, staging / SHARDS_DIR / shard_name(index), config.jpeg_quality)
            for index, source in enumerate(sources)
        ]
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=get_context("spawn")) as pool:
            tables = list(pool.map(_encode_job, jobs))
        columns = merge_tables(tables)
        savez: Any = np.savez  # numpy stubs type **arrays as allow_pickle
        savez(staging / INDEX_FILE, **columns)
        clips = []
        for index, (source, table) in enumerate(zip(sources, tables, strict=True)):
            manifest = source.manifest
            clips.append(
                {
                    "index": index,
                    "clip_id": source.clip_id,
                    "source_id": manifest.source.source_id,
                    "source_title": manifest.source.title,
                    "split": splits[manifest.source.source_id],
                    "width": manifest.width,
                    "height": manifest.height,
                    "frame_count": len(manifest.frames),
                    "time_base": manifest.time_base,
                    "nominal_fps": manifest.nominal_fps,
                    "track_ids": list(table.track_ids),
                    "annotation_status": source.annotation.status,
                    "annotation_issues": list(source.annotation.issues),
                    "annotation_sha256": source.annotation_sha256,
                    "video_sha256": manifest.sha256,
                    "stored_frames": int(len(table.frames["frame_index"])),
                }
            )
        stored_by_clip = [
            (splits[s.manifest.source.source_id], int(len(t.frames["frame_index"])))
            for s, t in zip(sources, tables, strict=True)
        ]
        sources_codes = columns["bbox_source"]
        metadata: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "version": config.version,
            "created_at": datetime.now(UTC).isoformat(),
            "annotation_root": str(config.annotation_root),
            "jpeg_quality": config.jpeg_quality,
            "split": {
                "unit": "source_id",
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "seed": config.split_seed,
            },
            "clips_without_players": without_players,
            "counts": {
                "clips": len(clips),
                "clips_without_players": len(without_players),
                "stored_frames": int(len(columns["frame_index"])),
                "annotated_frames": int(sum(len(s.annotation.frames) for s in sources)),
                "instances": int(len(sources_codes)),
                "instances_by_bbox_source": {
                    name: int((sources_codes == code).sum()) for name, code in BBOX_SOURCE_CODES.items()
                },
                "clips_by_split": {
                    split: sum(1 for clip in clips if clip["split"] == split) for split in SPLIT_CODES
                },
                "frames_by_split": {
                    split: sum(count for name, count in stored_by_clip if name == split)
                    for split in SPLIT_CODES
                },
            },
            "clips": clips,
        }
        (staging / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        (staging / "README.md").write_text(_summary_readme(metadata), encoding="utf-8")
        PlayerFrameStore(staging)  # Full read-side validation before publishing.
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
