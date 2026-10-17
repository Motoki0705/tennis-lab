"""Explicit ownership, retention, and read-only garbage-collection evidence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import (
    ArtifactRef,
    Digest,
    Document,
    Identifier,
    relative_path,
)
from ..contracts import (
    after_validator as _after_validator,
)

_Model = TypeVar("_Model")


class _AfterValidator(Protocol):
    def __call__(
        self, function: Callable[[_Model], _Model]
    ) -> Callable[[_Model], _Model]: ...


after_validator = cast(_AfterValidator, _after_validator)

ArtifactKind = Literal[
    "source",
    "clip",
    "cache",
    "attempt",
    "prediction",
    "adoption_evidence",
    "log",
    "shard",
    "tracking",
    "pose",
    "snapshot",
    "checkpoint",
    "configuration",
]


class Retention(Document):
    """No implicit TTL. Expiry only makes an unreferenced artifact eligible."""

    mode: Literal["retain", "expire"]
    expires_at_ns: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)

    @after_validator
    def explicit_expiry(self) -> Self:
        if (self.mode == "expire") != (self.expires_at_ns is not None):
            raise ValueError("only expire retention requires an explicit deadline")
        return self


class ManagedArtifact(Document):
    reference: ArtifactRef
    owner: Identifier
    stage: Identifier
    kind: ArtifactKind
    regenerable: bool
    retention: Retention
    dependencies: tuple[ArtifactRef, ...] = ()


class Protection(Document):
    protection_id: Identifier
    kind: Literal["adoption", "publication", "training", "model", "active", "reader"]
    reason: str = Field(min_length=1)
    references: tuple[ArtifactRef, ...] = Field(min_length=1)


class InventoryScope(Document):
    role: Literal["annotation", "data", "checkpoint", "artifact"]
    path: str

    @after_validator
    def valid_path(self) -> Self:
        relative_path(self.path)
        return self


class InventoryEntry(Document):
    path: str
    kind: Literal["file", "directory", "symlink", "special"]
    device: int
    inode: int
    links: int = Field(ge=1)
    logical_bytes: int = Field(ge=0)
    allocated_bytes: int = Field(ge=0)
    modified_ns: int
    changed_ns: int


class Inventory(Document):
    schema_version: Literal["annotation.storage.inventory.v1"] = (
        "annotation.storage.inventory.v1"
    )
    created_at_ns: int = Field(ge=0)
    scopes: tuple[InventoryScope, ...] = Field(min_length=1)
    entries: tuple[InventoryEntry, ...]
    logical_bytes: int = Field(ge=0)
    allocated_bytes: int = Field(ge=0)
    unique_inodes: int = Field(ge=0)
    files: int = Field(ge=0)
    symlinks: int = Field(ge=0)
    hardlinked_files: int = Field(ge=0)


class GCDecision(Document):
    path: str
    owner: str | None
    stage: str | None
    kind: ArtifactKind | None
    retention: Retention | None
    regenerable: bool | None
    eligible: bool
    reasons: tuple[str, ...] = Field(min_length=1)
    reference_count: int = Field(ge=0)


class StageUsage(Document):
    stage: str
    entries: int = Field(ge=0)
    files: int = Field(ge=0)
    symlinks: int = Field(ge=0)
    unique_inodes: int = Field(ge=0)
    hardlinked_files: int = Field(ge=0)
    logical_bytes: int = Field(ge=0)
    allocated_bytes: int = Field(ge=0)
    shared_inode_bytes: int = Field(ge=0)


class GCPlan(Document):
    schema_version: Literal["annotation.storage.gc-dry-run.v1"] = (
        "annotation.storage.gc-dry-run.v1"
    )
    created_at_ns: int = Field(ge=0)
    inventory_digest: Digest
    catalog_digest: Digest
    # The catalog is explicit, not a claim that every legacy reader is registered.
    execution_ready: Literal[False] = False
    decisions: tuple[GCDecision, ...]
    stages: tuple[StageUsage, ...]
    eligible_logical_bytes: int = Field(ge=0)
    eligible_allocated_bytes: int = Field(ge=0)
