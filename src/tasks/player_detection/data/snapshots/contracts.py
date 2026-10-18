"""An existing dataset is pinned explicitly; this is not a new annotation adoption."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    after_validator,
)
from src.tasks.player_detection.configuration import FrameSelectionConfig

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)
Split = Literal["train", "val", "test"]


class SourceBinding(Document):
    store_source_id: str = Field(min_length=1)
    source_id: Digest


class SnapshotClip(Document):
    clip_id: Digest
    store_clip_id: str = Field(min_length=1)
    source_id: Digest
    split: Split
    semantic_digest: Digest
    shard: ArtifactRef
    canonical_clip_id: Digest | None = None


class ExistingPlayerSnapshot(Document):
    schema_version: Literal["player.existing-store-snapshot.v1"] = (
        "player.existing-store-snapshot.v1"
    )
    index: ArtifactRef
    metadata: ArtifactRef
    source_bindings: tuple[SourceBinding, ...] = Field(min_length=1)
    clips: tuple[SnapshotClip, ...] = Field(min_length=1)
    # None denotes an explicitly frozen baseline. Publications pin their request.
    request: ArtifactRef | None = None

    @_after
    def identities(self) -> Self:
        if any(
            ref.role != "data"
            for ref in (self.index, self.metadata, *(c.shard for c in self.clips))
        ):
            raise ValueError("player snapshot files must be data artifacts")
        if len({c.clip_id for c in self.clips}) != len(self.clips) or len(
            {c.store_clip_id for c in self.clips}
        ) != len(self.clips):
            raise ValueError("player snapshot clip identities must be unique")
        keys = tuple(b.store_source_id for b in self.source_bindings)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("source bindings must be sorted and unique")
        return self


class SelectionPolicy(Document):
    require_reviewed: bool
    require_located_players: bool
    min_visible_box_px: float = Field(ge=0, le=1e4)
    frame_stride: int = Field(ge=1)

    def selection(self) -> FrameSelectionConfig:
        return FrameSelectionConfig(
            require_reviewed=self.require_reviewed,
            require_located_players=self.require_located_players,
            min_visible_box_px=self.min_visible_box_px,
        )


class PlayerPartition(Document):
    schema_version: Literal["player.partition.v1"] = "player.partition.v1"
    snapshot: ArtifactRef
    split: Split
    policy: SelectionPolicy
    selected_frames: int = Field(ge=1)
    selected_instances: int = Field(ge=0)
    negative_frames: int = Field(ge=0)
    # Ordered native frame identities and the exact clipped target boxes consumed.
    support_digest: Digest

    @_after
    def counts(self) -> Self:
        if (
            self.negative_frames > self.selected_frames
            or self.selected_instances < self.selected_frames - self.negative_frames
        ):
            raise ValueError("player partition support counts are inconsistent")
        if self.split != "train" and self.selected_instances == 0:
            raise ValueError(
                "fixed AP evaluation requires positive ground-truth instances"
            )
        return self
