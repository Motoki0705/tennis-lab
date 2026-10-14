"""A budget scope spans campaigns; changing campaign IDs cannot reset its counters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import ArtifactRef, Document, Identifier, PositiveInt, after_validator
from ..plan import ResourceBudget

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class BudgetLimits(Document):
    additional_storage_bytes: PositiveInt
    minimum_free_bytes: PositiveInt
    total_pss_bytes: PositiveInt
    gpu_seconds: PositiveInt

    @classmethod
    def from_budget(cls, budget: ResourceBudget) -> BudgetLimits:
        return cls(**{name: getattr(budget, name) for name in cls.model_fields})


class BudgetScope(Document):
    schema_version: Literal["annotation.resource-scope.v1"] = (
        "annotation.resource-scope.v1"
    )
    scope_id: Identifier
    limits: BudgetLimits
    roots: tuple[str, ...] = Field(min_length=1)
    baseline: ArtifactRef
    previously_allocated_bytes: int = Field(ge=0)
    previously_used_gpu_seconds: float = Field(ge=0)
    max_pending_per_stage: PositiveInt

    @_after
    def storage_boundaries(self) -> Self:
        paths = [Path(root) for root in self.roots]
        if any(not p.is_absolute() or str(p.resolve()) != str(p) for p in paths):
            raise ValueError(
                "storage roots must be normalized absolute paths without symlinks"
            )
        if any(
            a.is_relative_to(b)
            for i, a in enumerate(paths)
            for j, b in enumerate(paths)
            if i != j
        ):
            raise ValueError("storage roots must not overlap")
        return self


class ResourceRequest(Document):
    request_id: Identifier
    plan: ArtifactRef
    stage: Identifier
    pss_bytes: PositiveInt
    storage_bytes: int = Field(ge=0)
    gpu_seconds: float = Field(ge=0)


class ProcessUsage(Document):
    pid: PositiveInt
    start_ticks: PositiveInt
    group: PositiveInt
    session: PositiveInt
    pss_bytes: int = Field(ge=0)
    rss_bytes: int = Field(ge=0)
    read_bytes: int = Field(ge=0)
    write_bytes: int = Field(ge=0)


class StorageUsage(Document):
    allocated_bytes: int = Field(ge=0)
    additional_bytes: int = Field(ge=0)
    logical_bytes: int = Field(ge=0)
    entries: int = Field(ge=0)
    unique_inodes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)


class GPUUsage(Document):
    status: Literal["available", "unavailable"]
    owned_vram_bytes: int | None = Field(ge=0)
    device_vram_bytes: int | None = Field(ge=0)
    maximum_utilization_percent: float | None = Field(ge=0, le=100)
    reason: str | None


class ResourceSample(Document):
    measured_at: str
    storage_measured_at: str
    clock_boot_id: str | None = None
    measured_monotonic_ns: int | None = Field(default=None, ge=0)
    storage_measured_monotonic_ns: int | None = Field(default=None, ge=0)
    scope_id: Identifier
    storage: StorageUsage
    processes: tuple[ProcessUsage, ...]
    gpu: GPUUsage
    pss_bytes: int = Field(ge=0)
    rss_bytes: int = Field(ge=0)
    gpu_charged_seconds: float = Field(ge=0)
    gpu_active_seconds: float = Field(ge=0)
    gpu_reserved_seconds: float = Field(ge=0)
    active_by_stage: dict[str, int]
    pending_by_stage: dict[str, int]
    stopped_reason: str | None
