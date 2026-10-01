"""Build ``data/ball_detection/<version>`` from TrackNet, Meiji and chat-annotation balls.

Every source adapter returns validated :class:`ClipSpec` values. Each clip is
encoded to its own JPEG shard by a worker process; the index and metadata are
assembled afterwards and the whole version directory is published with one
atomic rename, so a partially built store is never visible.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    EVENT_CODES,
    FRAME_FIELDS,
    INDEX_FILE,
    INSTANCE_FIELDS,
    INSTANCE_XY,
    METADATA_FILE,
    POINT_KIND_CODES,
    SCHEMA_VERSION,
    SHARDS_DIR,
    SOURCES,
    SPLIT_CODES,
    BallFrameStore,
    Split,
    shard_name,
    split_names,
)
from src.tasks.ball_detection.generate_dataset.frame_store.clip import ClipSpec
from src.tasks.ball_detection.generate_dataset.frame_store.config import (
    ExplicitGroupSplit,
    FrameStoreBuildConfig,
    SourceSplit,
)
from src.tasks.ball_detection.generate_dataset.frame_store.sources import (
    collect_chat_annotation,
    collect_meiji,
    collect_tracknet,
)
from src.utils.data.splits import make_group_split_map


@dataclass(frozen=True, slots=True)
class StoredSize:
    """Stored image size of a clip and the source->stored pixel factor."""

    width: int
    height: int

    @classmethod
    def for_clip(cls, width: int, height: int, max_height: int) -> StoredSize:
        if height <= max_height:
            return cls(width, height)
        scaled_width = width * max_height / height
        if scaled_width != int(scaled_width):
            raise ValueError(
                f"{width}x{height} cannot be downscaled to height {max_height} without "
                "changing the aspect ratio; choose another dataset.max_height"
            )
        return cls(int(scaled_width), max_height)


@dataclass(frozen=True, slots=True)
class EncodedClip:
    """Frame/instance columns of one clip; offsets are relative to its shard."""

    frames: dict[str, NDArray[np.generic]]
    instances: dict[str, NDArray[np.generic]]


def encode_clip(spec: ClipSpec, shard_path: Path, jpeg_quality: int, size: StoredSize) -> EncodedClip:
    """Write every frame of the clip, in order, and scale its labels to stored pixels.

    A JPEG source that needs no resize is stored byte-for-byte (after checking it
    decodes to the declared size); everything else is (resized and) re-encoded.
    """
    resize = (size.width, size.height) != (spec.width, spec.height)
    offsets: list[int] = []
    lengths: list[int] = []
    offset = 0
    with shard_path.open("xb") as shard:
        for index, frame in enumerate(spec.media.iter_frames()):
            if isinstance(frame, bytes) and not resize:
                decoded = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape != (spec.height, spec.width, 3):
                    raise ValueError(f"{spec.clip_id}: frame {index} is not a {spec.width}x{spec.height} JPEG")
                payload = frame
            else:
                pixels = (
                    cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if isinstance(frame, bytes)
                    else frame
                )
                if pixels is None or pixels.shape != (spec.height, spec.width, 3):
                    raise ValueError(f"{spec.clip_id}: frame {index} does not decode to {spec.width}x{spec.height}")
                if resize:
                    pixels = cv2.resize(pixels, (size.width, size.height), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if not ok:
                    raise RuntimeError(f"{spec.clip_id}: JPEG encoding failed at frame {index}")
                payload = encoded.tobytes()
            shard.write(payload)
            offsets.append(offset)
            lengths.append(len(payload))
            offset += len(payload)
    labels = spec.labels
    if len(offsets) != labels.frame_count:
        raise ValueError(f"{spec.clip_id}: stored {len(offsets)} frames for {labels.frame_count} labels")
    frames = {name: labels.frames[name] for name in labels.frames}
    frames["frame_index"] = np.arange(labels.frame_count, dtype=np.int32)
    frames["offset"] = np.asarray(offsets, dtype=np.int64)
    frames["length"] = np.asarray(lengths, dtype=np.int64)
    instances = dict(labels.instances)
    # One factor for both axes (StoredSize keeps the aspect ratio exactly); this
    # is the corner-anchored ``x * W'/W`` convention used by the datasets.
    instances[INSTANCE_XY] = (labels.xy * (size.width / spec.width)).astype(np.float32)
    return EncodedClip(frames, instances)


def _encode_job(job: tuple[ClipSpec, Path, int, StoredSize]) -> EncodedClip:
    return encode_clip(*job)


def assign_splits(specs: list[ClipSpec], split: SourceSplit) -> dict[str, Split]:
    """Map every group of one source to a split; explicit maps must match exactly."""
    groups = sorted({spec.group_id for spec in specs})
    if isinstance(split, ExplicitGroupSplit):
        missing = sorted(set(groups) - set(split.groups))
        stale = sorted(set(split.groups) - set(groups))
        if missing or stale:
            raise ValueError(f"Split groups differ from the data: unassigned {missing}, absent {stale}")
        return {group: split.groups[group] for group in groups}
    # Hashed group split weighted by frames with an observed ball.
    weights: Counter[str] = Counter()
    for spec in specs:
        weights[spec.group_id] += int(
            np.count_nonzero(spec.labels.instances["point_kind"] == POINT_KIND_CODES["observed"])
        )
    assigned = make_group_split_map(dict(sorted(weights.items())), split)
    return {group: _split_name(assigned[group]) for group in groups}


def _split_name(value: str) -> Split:
    (split,) = split_names([value])
    return split


def collect_sources(config: FrameStoreBuildConfig) -> list[tuple[ClipSpec, Split]]:
    """All configured sources in the fixed order of ``SOURCES``, with their splits."""
    collected: list[tuple[ClipSpec, Split]] = []
    specs: list[ClipSpec]
    split: SourceSplit
    for name in SOURCES:
        if name == "tracknet" and config.tracknet is not None:
            specs, split = collect_tracknet(config.tracknet[0]), config.tracknet[1]
        elif name == "meiji" and config.meiji is not None:
            specs, split = collect_meiji(config.meiji[0]), config.meiji[1]
        elif name == "chat_annotation" and config.chat_annotation is not None:
            specs, split = collect_chat_annotation(config.chat_annotation[0]), config.chat_annotation[1]
        else:
            continue
        splits = assign_splits(specs, split)
        collected.extend((spec, splits[spec.group_id]) for spec in specs)
    return collected


def merge_tables(tables: list[EncodedClip]) -> dict[str, NDArray[np.generic]]:
    """Concatenate clip tables, setting clip indices and global instance starts."""
    columns: dict[str, list[NDArray[np.generic]]] = {
        name: [] for name in (*FRAME_FIELDS, *INSTANCE_FIELDS, INSTANCE_XY) if name not in {"clip", "inst_start"}
    }
    clip_column: list[NDArray[np.int32]] = []
    for clip_index, table in enumerate(tables):
        for name in columns:
            columns[name].append(table.frames[name] if name in table.frames else table.instances[name])
        clip_column.append(np.full(len(table.frames["frame_index"]), clip_index, dtype=np.int32))
    merged = {name: np.concatenate(values) for name, values in columns.items()}
    merged["clip"] = np.concatenate(clip_column)
    counts = merged["inst_count"].astype(np.int64)
    merged["inst_start"] = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    for name, dtype in {**FRAME_FIELDS, **INSTANCE_FIELDS}.items():
        merged[name] = merged[name].astype(dtype, copy=False)
    merged[INSTANCE_XY] = merged[INSTANCE_XY].reshape(-1, 2).astype(np.float32, copy=False)
    return merged


def _counts(clips: list[dict[str, Any]], columns: Mapping[str, NDArray[np.generic]]) -> dict[str, object]:
    frame_source = np.asarray([clips[int(c)]["source"] for c in columns["clip"]])
    instance_source: NDArray[np.str_] = np.repeat(frame_source, columns["inst_count"].astype(np.int64))
    kinds = columns["point_kind"]
    annotated = columns["annotated"].astype(bool)
    empty = columns["inst_count"] == 0
    by_source: dict[str, object] = {}
    for source in sorted(set(frame_source.tolist())):
        in_source = frame_source == source
        by_source[source] = {
            "clips": sum(1 for clip in clips if clip["source"] == source),
            "frames": int(in_source.sum()),
            "annotated_frames": int((in_source & annotated).sum()),
            "negative_frames": int((in_source & annotated & empty).sum()),
            "segment_breaks": int((in_source & columns["segment_break"].astype(bool)).sum()),
            "instances_by_point_kind": {
                name: int(((instance_source == source) & (kinds == code)).sum())
                for name, code in POINT_KIND_CODES.items()
            },
            "events": {
                name: int((in_source & (columns["event"] == code)).sum()) for name, code in EVENT_CODES.items()
            },
            "clips_by_split": {
                split: sum(1 for clip in clips if clip["source"] == source and clip["split"] == split)
                for split in SPLIT_CODES
            },
            "frames_by_split": {
                split: sum(
                    clip["frame_count"] for clip in clips if clip["source"] == source and clip["split"] == split
                )
                for split in SPLIT_CODES
            },
            "groups_by_split": {
                split: sorted({clip["group_id"] for clip in clips if clip["source"] == source and clip["split"] == split})
                for split in SPLIT_CODES
            },
        }
    return {"clips": len(clips), "frames": int(len(columns["clip"])), "by_source": by_source}


def _summary_readme(metadata: dict[str, Any]) -> str:
    rows = []
    for source, counts in metadata["counts"]["by_source"].items():
        kinds = counts["instances_by_point_kind"]
        frames = counts["frames_by_split"]
        rows.append(
            f"| {source} | {counts['clips']} | {counts['frames']} | "
            f"{frames['train']} / {frames['val']} / {frames['test']} | {counts['negative_frames']} | "
            + " | ".join(str(kinds[name]) for name in POINT_KIND_CODES)
            + " |"
        )
    header = "| source | clips | frames | frames train / val / test | negative frames | " + " | ".join(
        POINT_KIND_CODES
    ) + " |"
    return (
        f"# ball_detection dataset `{metadata['version']}`\n\n"
        "Generated by `python -m src.tasks.ball_detection.scripts.generate_dataset`.\n"
        "Schema, label semantics and access API: `src/tasks/ball_detection/data/store.py`.\n\n"
        f"{header}\n|" + "---|" * (5 + len(POINT_KIND_CODES)) + "\n" + "\n".join(rows) + "\n\n"
        "Split groups and every other count are in `metadata.json`.\n"
    )


def build_dataset(config: FrameStoreBuildConfig) -> Path:
    """Build and atomically publish one dataset version; returns its directory."""
    destination = config.dataset_dir
    if destination.exists():
        raise FileExistsError(f"Dataset version already exists: {destination}. Choose a new dataset.version.")
    collected = collect_sources(config)
    ids = [spec.clip_id for spec, _ in collected]
    if len(ids) != len(set(ids)):
        raise ValueError("Ball sources produced duplicate clip ids")
    sizes = [StoredSize.for_clip(spec.width, spec.height, config.max_height) for spec, _ in collected]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".building-{destination.name}-{uuid.uuid4().hex[:8]}"
    (staging / SHARDS_DIR).mkdir(parents=True)
    try:
        jobs = [
            (spec, staging / SHARDS_DIR / shard_name(index), config.jpeg_quality, size)
            for index, ((spec, _), size) in enumerate(zip(collected, sizes, strict=True))
        ]
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=get_context("spawn")) as pool:
            tables = list(pool.map(_encode_job, jobs))
        columns = merge_tables(tables)
        savez: Any = np.savez  # numpy stubs type **arrays as allow_pickle
        savez(staging / INDEX_FILE, **columns)
        clips: list[dict[str, Any]] = [
            {
                "index": index,
                "clip_id": spec.clip_id,
                "source": spec.source,
                "group_id": spec.group_id,
                "camera_id": spec.camera_id,
                "split": split,
                "width": size.width,
                "height": size.height,
                "source_width": spec.width,
                "source_height": spec.height,
                "frame_count": spec.labels.frame_count,
                "time_base": str(spec.time_base),
                "fps": str(spec.fps),
                "has_events": spec.has_events,
                "track_ids": list(spec.labels.track_ids),
                "annotation_path": str(spec.annotation_path),
                "annotation_sha256": spec.annotation_sha256,
                "media_sha256": spec.media_sha256,
                "provenance": spec.provenance,
            }
            for index, ((spec, split), size) in enumerate(zip(collected, sizes, strict=True))
        ]
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "version": config.version,
            "created_at": datetime.now(UTC).isoformat(),
            "jpeg_quality": config.jpeg_quality,
            "max_height": config.max_height,
            "sources": _source_settings(config),
            "counts": _counts(clips, columns),
            "clips": clips,
        }
        (staging / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        (staging / "README.md").write_text(_summary_readme(metadata), encoding="utf-8")
        BallFrameStore(staging)  # Full read-side validation before publishing.
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _source_settings(config: FrameStoreBuildConfig) -> dict[str, object]:
    settings: dict[str, object] = {}
    if config.tracknet is not None:
        source, split = config.tracknet
        settings["tracknet"] = {"root": str(source.root), "fps": str(source.fps), "split": dict(split.groups)}
    if config.meiji is not None:
        meiji, meiji_split = config.meiji
        settings["meiji"] = {"root": str(meiji.root), "split": dict(meiji_split.groups)}
    if config.chat_annotation is not None:
        chat, chat_split = config.chat_annotation
        settings["chat_annotation"] = {
            "annotation_root": str(chat.annotation_root),
            "allowed_statuses": sorted(chat.allowed_statuses),
            "split": {
                "unit": "source_id",
                "val_ratio": chat_split.val_ratio,
                "test_ratio": chat_split.test_ratio,
                "seed": chat_split.seed,
            },
        }
    return settings
