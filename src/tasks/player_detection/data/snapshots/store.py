"""Freeze existing rows, native timelines, teacher semantics and RGB file identities."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.tasks.base.annotation.artifacts import (
    data_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    canonical_bytes,
    content_digest,
)
from src.tasks.base.annotation.paths import AnnotationLayout

from ..store import (
    INDEX_FILE,
    METADATA_FILE,
    SHARDS_DIR,
    ClipRecord,
    PlayerFrameStore,
    shard_name,
)
from .contracts import ExistingPlayerSnapshot, SnapshotClip, SourceBinding


def clip_identity(clip: ClipRecord, source_id: str) -> str:
    result: str = content_digest(
        ["legacy_player_clip", source_id, clip.clip_id, clip.video_sha256]
    )
    return result


def clip_semantic_digest(store: PlayerFrameStore, clip: ClipRecord) -> str:
    """Global row positions and v1/v2 encoding are not annotation semantics."""
    header = asdict(clip)
    for field in ("index", "split"):
        header.pop(field)
    digest = hashlib.sha256(canonical_bytes(header))
    rows = np.flatnonzero(store.frames["clip"] == clip.index)
    for name, column in sorted(store.frames.items()):
        if name not in {"clip", "inst_start"}:
            selected = np.ascontiguousarray(column[rows])
            digest.update(
                canonical_bytes(["frame", name, selected.dtype.str, selected.shape])
            )
            digest.update(selected.tobytes())
    instances = np.concatenate(
        [
            np.arange(
                int(store.frames["inst_start"][row]),
                int(store.frames["inst_start"][row] + store.frames["inst_count"][row]),
                dtype=np.int64,
            )
            for row in rows
        ]
    )
    for name, column in sorted(store.instances.items()):
        selected = np.ascontiguousarray(column[instances])
        if np.issubdtype(selected.dtype, np.floating):
            # Canonicalize NaN payload bits; unresolved boxes remain unresolved.
            selected[np.isnan(selected)] = np.nan
        digest.update(
            canonical_bytes(["instance", name, selected.dtype.str, selected.shape])
        )
        digest.update(selected.tobytes())
    return digest.hexdigest()


def _check_clip(store: PlayerFrameStore, clip: ClipRecord) -> None:
    rows = np.flatnonzero(store.frames["clip"] == clip.index)
    if (
        min(clip.width, clip.height, clip.frame_count) < 1
        or Fraction(clip.time_base) <= 0
        or Fraction(clip.nominal_fps) <= 0
    ):
        raise ValueError("player snapshot requires positive native geometry/timebase")
    if len(set(clip.track_ids)) != len(clip.track_ids):
        raise ValueError("player snapshot track identities must be unique")
    for name in ("frame_index", "source_frame_index", "clip_pts"):
        values = store.frames[name][rows]
        if np.any(values < 0) or np.any(np.diff(values) <= 0):
            raise ValueError("player snapshot native frame/PTS sequence must increase")
    offsets, lengths = store.frames["offset"][rows], store.frames["length"][rows]
    if (
        np.any(lengths <= 0)
        or offsets[0] != 0
        or not np.array_equal(offsets[1:], np.cumsum(lengths)[:-1])
    ):
        raise ValueError(
            "player snapshot frame byte ranges must exactly partition its shard"
        )


def validate_snapshot(
    layout: AnnotationLayout, snapshot: ExistingPlayerSnapshot
) -> PlayerFrameStore:
    directory = layout.artifact_path(snapshot.index).parent
    if (
        layout.artifact_path(snapshot.index),
        layout.artifact_path(snapshot.metadata),
    ) != (directory / INDEX_FILE, directory / METADATA_FILE):
        raise ValueError(
            "player snapshot index/metadata must share the exact store directory"
        )
    files = (snapshot.index, snapshot.metadata, *(c.shard for c in snapshot.clips))
    for ref in files:
        verify_reference(layout, ref)
    store = PlayerFrameStore(directory)
    mapping = {b.store_source_id: b.source_id for b in snapshot.source_bindings}
    if set(mapping) != {c.source_id for c in store.clips} or len(store.clips) != len(
        snapshot.clips
    ):
        raise ValueError(
            "player snapshot source/clip membership differs from its store"
        )
    splits: dict[str, str] = {}
    for clip, identity in zip(store.clips, snapshot.clips, strict=True):
        _check_clip(store, clip)
        source = mapping[clip.source_id]
        if (
            identity.clip_id,
            identity.store_clip_id,
            identity.source_id,
            identity.split,
            identity.semantic_digest,
        ) != (
            clip_identity(clip, source),
            clip.clip_id,
            source,
            clip.split,
            clip_semantic_digest(store, clip),
        ):
            raise ValueError(
                "player snapshot clip labels, native timeline, identity or split changed"
            )
        if layout.artifact_path(identity.shard) != directory / SHARDS_DIR / shard_name(
            clip.index
        ):
            raise ValueError("player snapshot clip points to another RGB shard")
        if splits.setdefault(source, clip.split) != clip.split:
            raise ValueError("one canonical player source crosses dataset partitions")
    # A mutation during reader construction/digesting must not produce a snapshot.
    for ref in files:
        verify_reference(layout, ref)
    return store


def load_snapshot(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ExistingPlayerSnapshot, PlayerFrameStore]:
    snapshot = read_document(layout, reference, ExistingPlayerSnapshot)
    return snapshot, validate_snapshot(layout, snapshot)


def freeze_existing_store(
    layout: AnnotationLayout, directory: Path, bindings: tuple[SourceBinding, ...]
) -> ArtifactRef:
    index, metadata = (
        data_reference(layout, directory / INDEX_FILE),
        data_reference(layout, directory / METADATA_FILE),
    )
    store = PlayerFrameStore(directory)
    mapping = {b.store_source_id: b.source_id for b in bindings}
    if len(mapping) != len(bindings) or set(mapping) != {
        c.source_id for c in store.clips
    }:
        raise ValueError(
            "existing player source bindings must cover every stored source exactly once"
        )
    snapshot = ExistingPlayerSnapshot(
        index=index,
        metadata=metadata,
        source_bindings=tuple(sorted(bindings, key=lambda b: b.store_source_id)),
        clips=tuple(
            SnapshotClip(
                clip_id=clip_identity(c, mapping[c.source_id]),
                store_clip_id=c.clip_id,
                source_id=mapping[c.source_id],
                split=c.split,
                semantic_digest=clip_semantic_digest(store, c),
                shard=data_reference(
                    layout, directory / SHARDS_DIR / shard_name(c.index)
                ),
            )
            for c in store.clips
        ),
    )
    validate_snapshot(layout, snapshot)
    result: ArtifactRef = write_document(
        layout,
        layout.path(
            "campaigns", "imported-player-snapshots", snapshot.sha256 + ".json"
        ),
        snapshot,
    )
    return result
