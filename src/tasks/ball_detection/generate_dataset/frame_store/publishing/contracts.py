"""Pinned snapshots, explicit source splits and publication requests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    FrameRange,
    Identifier,
    after_validator,
)

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)
Split = Literal["train", "val", "test"]


class SourceSplit(Document):
    source_id: Digest
    split: Split


class SourcePartition(Document):
    """Freeze an ID-only allocation rule before collecting replacement sources."""

    method: Literal["ranked_source_hash"] = "ranked_source_hash"
    seed: str = Field(min_length=1)
    train_weight: int = Field(gt=0)
    validation_weight: int = Field(gt=0)
    test_weight: int = Field(gt=0)


class BallPublicationPolicy(Document):
    schema_version: Literal["ball.publication-policy.v1"] = "ball.publication-policy.v1"
    jpeg_quality: int = Field(ge=1, le=100)
    image_storage: Literal["jpeg", "video_q90"] = "jpeg"
    source_splits: tuple[SourceSplit, ...]
    source_partition: SourcePartition | None = None

    @_after
    def unique_sources(self) -> Self:
        if self.source_partition is not None and self.source_splits:
            raise ValueError("Choose explicit source splits or a source partition recipe")
        if self.image_storage == "video_q90" and self.jpeg_quality != 90:
            raise ValueError("video_q90 requires the unchanged quality-90 JPEG contract")
        if len({s.source_id for s in self.source_splits}) != len(self.source_splits):
            raise ValueError("publication source splits must be unique")
        return self


class LegacySourceGroup(Document):
    """Explicitly bridge old source/group names to fixed-evaluation identities."""

    source: Literal["tracknet", "meiji", "chat_annotation"]
    group_id: str = Field(min_length=1)
    source_id: Digest


class LegacyClipRange(Document):
    """Explicit source-frame interval for an imported chat-annotation clip."""

    store_clip_id: str = Field(min_length=1)
    source_range: FrameRange


class SnapshotClip(Document):
    clip_id: Digest
    canonical_clip_id: Digest | None
    store_clip_id: str = Field(min_length=1)
    source_id: Digest
    split: Split
    semantic_digest: Digest
    shard: ArtifactRef


class BallSnapshot(Document):
    schema_version: Literal["ball.snapshot.v1"] = "ball.snapshot.v1"
    index: ArtifactRef
    metadata: ArtifactRef
    clips: tuple[SnapshotClip, ...] = Field(min_length=1)
    # None is an explicitly imported v1 baseline, never an adopted snapshot.
    request: ArtifactRef | None
    legacy_groups: tuple[LegacySourceGroup, ...]

    @_after
    def identities(self) -> Self:
        if (
            self.index.role != "data"
            or self.metadata.role != "data"
            or any(c.shard.role != "data" for c in self.clips)
        ):
            raise ValueError(
                "snapshot index, metadata and shards must be data artifacts"
            )
        if len({c.clip_id for c in self.clips}) != len(self.clips) or len(
            {c.store_clip_id for c in self.clips}
        ) != len(self.clips):
            raise ValueError("snapshot clip identities must be unique")
        canonical_ids = [c.canonical_clip_id for c in self.clips if c.canonical_clip_id]
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("snapshot canonical clip identities must be unique")
        if bool(self.legacy_groups) != (self.request is None):
            raise ValueError(
                "snapshot must declare either a legacy import or a publication request"
            )
        return self


class PublishRequest(Document):
    schema_version: Literal["ball.publication-request.v1"] = (
        "ball.publication-request.v1"
    )
    plan: ArtifactRef
    product: Identifier
    adoptions: tuple[ArtifactRef, ...] = Field(min_length=1)

    @_after
    def unique_adoptions(self) -> Self:
        digests = [a.digest for a in self.adoptions]
        if digests != sorted(set(digests)):
            raise ValueError("adoption receipts must be unique and sorted by digest")
        return self
