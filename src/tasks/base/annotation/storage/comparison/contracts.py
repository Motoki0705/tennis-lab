"""Pinned inputs and evidence for equal-input format/copy/cache measurements."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from ...contracts import ArtifactRef, Digest, Document, Identifier, PositiveInt
from ..contracts import InventoryScope


class StoreSpec(Document):
    adapter: Identifier
    root: InventoryScope
    descriptor: dict[str, JsonValue] | None = None


class ReadLimits(Document):
    maximum_entries: PositiveInt
    maximum_source_bytes: PositiveInt
    maximum_frames: PositiveInt
    maximum_pixels_per_frame: PositiveInt


class StoreFile(Document):
    relative: str
    reference: ArtifactRef
    shared_shard: bool


class CapturedStore(Document):
    schema_version: Literal["annotation.storage.capture.v1"] = (
        "annotation.storage.capture.v1"
    )
    specification: ArtifactRef
    files: tuple[StoreFile, ...] = Field(min_length=1)
    frame_count: PositiveInt
    metadata_digest: Digest


class ComparisonRequest(Document):
    source: ArtifactRef
    rows: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=1)
    attempts: int = Field(ge=1, le=50)
    passes: int = Field(ge=2, le=100)
    limits: ReadLimits
    # Conservative reservation of newly written file payloads; resource scope
    # separately bounds allocated storage, free bytes, and process-tree PSS.
    maximum_written_bytes: PositiveInt
    converted_shard_bytes: PositiveInt


class RowEvidence(Document):
    row: int = Field(ge=0)
    key: str = Field(min_length=1)
    label_digest: Digest
    jpeg_digest: Digest
    rgb_digest: Digest
    shape: tuple[PositiveInt, PositiveInt, Literal[3]]


class PassTiming(Document):
    read_ns: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    decoded_frames: int = Field(ge=0)


class AttemptTiming(Document):
    clone_ns: int = Field(ge=0)
    open_ns: int = Field(ge=0)
    passes: tuple[PassTiming, ...]


class PolicyResult(Document):
    format: Literal["native", "image_records"]
    shards: Literal["copy", "share"]
    cache: Literal["decode", "private_rgb", "shared_rgb"]
    inventory: ArtifactRef
    attempts: tuple[AttemptTiming, ...]


class ComparisonReport(Document):
    schema_version: Literal["annotation.storage.comparison.v1"] = (
        "annotation.storage.comparison.v1"
    )
    request: ArtifactRef
    decoder: str
    source_inventory: ArtifactRef
    converted_inventory: ArtifactRef
    conversion_ns: int = Field(ge=0)
    all_rows_digest: Digest
    rows: tuple[RowEvidence, ...]
    policies: tuple[PolicyResult, ...]
    reserved_written_bytes: int = Field(ge=0)
    adopted_policy: None = None
