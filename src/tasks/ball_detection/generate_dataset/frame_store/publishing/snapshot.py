"""Validate all file identities before exposing a published store to a consumer."""

from __future__ import annotations

from pathlib import Path

from src.tasks.ball_detection.data.store import (
    SCHEMA_VERSION,
    BallFrameStore,
    shard_name,
)
from src.tasks.ball_detection.data.windows.manifest import clip_semantic_digest
from src.tasks.base.annotation.artifacts import (
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, content_digest
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import file_digest

from .contracts import BallSnapshot, LegacySourceGroup, SnapshotClip


def data_reference(layout: AnnotationLayout, path: Path) -> ArtifactRef:
    relative = path.relative_to(layout.repository / "data").as_posix()
    # Validate the exact path (including symlinks) before hashing it.
    layout.artifact_path(
        ArtifactRef(role="data", path=relative, digest="0" * 64, size_bytes=0)
    )
    result = ArtifactRef(
        role="data",
        path=relative,
        digest=file_digest(path),
        size_bytes=path.stat().st_size,
    )
    return result


def validate_snapshot(
    layout: AnnotationLayout,
    snapshot: BallSnapshot,
    *,
    staging: Path | None = None,
) -> BallFrameStore:
    """Staging is only used by the publisher, with final paths pinned in the manifest."""
    destination = layout.artifact_path(snapshot.index).parent
    if (
        snapshot.index.path
        != snapshot.metadata.path.removesuffix("metadata.json") + "index.npz"
    ):
        raise ValueError("snapshot index/metadata must share the store directory")
    if Path(snapshot.metadata.path).name != "metadata.json":
        raise ValueError("snapshot metadata filename is invalid")
    for reference in (
        snapshot.index,
        snapshot.metadata,
        *(c.shard for c in snapshot.clips),
    ):
        source = layout.artifact_path(reference)
        if not source.is_relative_to(destination):
            raise ValueError("snapshot file escapes its store directory")
        if staging is None:
            verify_reference(layout, reference)
        else:
            candidate = staging / source.relative_to(destination)
            # The same data-root/symlink boundary applies to in-progress files.
            if data_reference(layout, candidate).model_dump(
                exclude={"path"}
            ) != reference.model_dump(exclude={"path"}):
                raise ValueError("staging snapshot file hash or size changed")
    store = BallFrameStore(staging or destination)
    if len(store.clips) != len(snapshot.clips):
        raise ValueError("snapshot membership differs from the store")
    splits: dict[str, str] = {}
    for clip, identity in zip(store.clips, snapshot.clips, strict=True):
        if (clip.clip_id, clip.split, clip_semantic_digest(store, clip)) != (
            identity.store_clip_id,
            identity.split,
            identity.semantic_digest,
        ):
            raise ValueError("snapshot clip labels, timeline or split changed")
        expected = destination / "shards" / shard_name(clip.index)
        if layout.artifact_path(identity.shard) != expected:
            raise ValueError("snapshot clip refers to the wrong RGB shard")
        if splits.setdefault(identity.source_id, identity.split) != identity.split:
            raise ValueError("snapshot splits one source across partitions")
    return store


def load_snapshot(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[BallSnapshot, BallFrameStore]:
    snapshot = read_document(layout, reference, BallSnapshot)
    return snapshot, validate_snapshot(layout, snapshot)


def freeze_legacy_store(
    layout: AnnotationLayout,
    directory: Path,
    groups: tuple[LegacySourceGroup, ...],
) -> ArtifactRef:
    """Pin a v1 baseline without rewriting it or inventing canonical source IDs."""
    index = data_reference(layout, directory / "index.npz")
    metadata = data_reference(layout, directory / "metadata.json")
    store = BallFrameStore(directory)
    if store.metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "only a v1 baseline may be imported; reuse an adopted snapshot manifest"
        )
    mapping = {(g.source, g.group_id): g.source_id for g in groups}
    if len(mapping) != len(groups) or set(mapping) != {
        (c.source, c.group_id) for c in store.clips
    }:
        raise ValueError(
            "legacy source bindings must cover each stored group exactly once"
        )
    snapshot = BallSnapshot(
        index=index,
        metadata=metadata,
        request=None,
        legacy_groups=tuple(sorted(groups, key=lambda g: (g.source, g.group_id))),
        clips=tuple(
            SnapshotClip(
                clip_id=content_digest(["legacy_ball_clip", c.source, c.clip_id]),
                store_clip_id=c.clip_id,
                source_id=mapping[c.source, c.group_id],
                split=c.split,
                semantic_digest=clip_semantic_digest(store, c),
                shard=data_reference(
                    layout, directory / "shards" / shard_name(c.index)
                ),
            )
            for c in store.clips
        ),
    )
    validate_snapshot(layout, snapshot)
    result: ArtifactRef = write_document(
        layout,
        layout.path("campaigns", "imported-ball-snapshots", snapshot.sha256 + ".json"),
        snapshot,
    )
    return result
