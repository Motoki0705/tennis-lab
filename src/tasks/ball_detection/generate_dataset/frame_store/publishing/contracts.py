"""Pinned snapshots, explicit source splits and publication requests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar

from pydantic import Field, SerializerFunctionWrapHandler

from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    FrameRange,
    Identifier,
    after_validator,
    manifest_serializer,
)
from src.tasks.base.annotation.plan import ProductPlan

Split = Literal["train", "val", "test"]
_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


class _Serializer(Protocol):
    def __call__(self, function: Callable[[_M, SerializerFunctionWrapHandler], dict[str, object]]) -> Callable[[_M, SerializerFunctionWrapHandler], dict[str, object]]: ...


_after: _Validator = after_validator
_serialize: _Serializer = manifest_serializer


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
    target_semantics: Literal["legacy_ball", "active_ball"] = "legacy_ball"
    position_weighting: Literal["uniform_observed_and_estimated"] = (
        "uniform_observed_and_estimated"
    )

    @_serialize
    def serialize_policy(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if self.target_semantics == "legacy_ball":
            result.pop("target_semantics", None)
            result.pop("position_weighting", None)
        return result

    @_after
    def unique_sources(self) -> Self:
        if self.source_partition is not None and self.source_splits:
            raise ValueError("Choose explicit source splits or a source partition recipe")
        if self.image_storage == "video_q90" and self.jpeg_quality != 90:
            raise ValueError("video_q90 requires the unchanged quality-90 JPEG contract")
        if self.target_semantics == "active_ball" and (
            self.image_storage != "jpeg" or self.jpeg_quality != 90
        ):
            raise ValueError("active_ball requires unchanged prepared Q90 JPEG images")
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


class SavedVideoAnnotation(Document):
    """A saved v3 revision and its independently registered blank input."""

    context: ArtifactRef
    annotation: ArtifactRef


class VideoPublishRequest(Document):
    """Explicit output selection for a complete local annotation plan.

    LocalAnnotationPlan deliberately has no model, learning recipe or products.
    Keep the dataset version/policy here, separate from the annotation schema.
    """

    schema_version: Literal["ball.video-publication-request.v1"] = (
        "ball.video-publication-request.v1"
    )
    plan: ArtifactRef
    output: ProductPlan
    annotations: tuple[SavedVideoAnnotation, ...] = Field(min_length=1)

    @property
    def product(self) -> str:
        return str(self.output.product)

    @property
    def adoptions(self) -> tuple[ArtifactRef, ...]:
        # Completed v3 judgments are trusted directly, without a legacy QA seal.
        return ()

    @_after
    def unique_annotations(self) -> Self:
        keys = [a.context.digest for a in self.annotations]
        if keys != sorted(set(keys)):
            raise ValueError("video annotation contexts must be unique and digest sorted")
        return self


class MigratedVideoAnnotation(Document):
    """A converter result, pinned decoder input, original v2, and native images."""

    expected_input: ArtifactRef
    annotation: ArtifactRef
    original: ArtifactRef
    migration: ArtifactRef
    manifest: ArtifactRef
    source_media: ArtifactRef


class MigrationPublishRequest(Document):
    """Publish the exact selected migration set without creating annotation work.

    The immutable request is the complete input selection. It has no worker,
    execution/resource budget, or implicit collection of unrelated inputs.
    """

    schema_version: Literal["ball.migration-publication-request.v1"] = (
        "ball.migration-publication-request.v1"
    )
    output: ProductPlan
    annotations: tuple[MigratedVideoAnnotation, ...] = Field(min_length=1)

    @property
    def product(self) -> str:
        return str(self.output.product)

    @property
    def adoptions(self) -> tuple[ArtifactRef, ...]:
        return ()

    @_after
    def unique_annotations(self) -> Self:
        keys = [a.expected_input.digest for a in self.annotations]
        if keys != sorted(set(keys)):
            raise ValueError("migration inputs must be unique and digest sorted")
        return self
