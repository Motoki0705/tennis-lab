"""Measurements never turn configured capacity or synthetic workers into real load."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import ArtifactRef, Document, Identifier, after_validator
from ..resources.contracts import ResourceSample

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class WorkerBinding(Document):
    stage: Identifier
    adapter: Identifier
    adapter_version: Identifier
    task: Identifier


class ConditionSpec(Document):
    benchmark_id: Identifier
    condition_id: Identifier
    plan: ArtifactRef
    scope: ArtifactRef
    predictions: tuple[ArtifactRef, ...] = Field(min_length=1, max_length=10000)
    bindings: tuple[WorkerBinding, ...] = Field(min_length=1)
    target_workers: Literal[10, 25, 50]
    maximum_observations: int = Field(ge=1, le=100000)

    @_after
    def distinct_bindings(self) -> Self:
        if len({b.adapter for b in self.bindings}) != len(self.bindings):
            raise ValueError("each measured adapter needs exactly one binding")
        if len(set(self.predictions)) != len(self.predictions):
            raise ValueError("prediction receipts must be unique")
        return self


class ComparisonSpec(Document):
    benchmark_id: Identifier
    conditions: tuple[ArtifactRef, ...] = Field(min_length=3, max_length=3)
    # Task-owned fixed-example evaluation. Merely attaching a file is not a pass.
    quality_policy: ArtifactRef | None
    quality_evidence: tuple[ArtifactRef, ...] = ()


class AttemptSnapshot(Document):
    launch: ArtifactRef
    state: str
    execution: ArtifactRef | None
    telemetry: ArtifactRef | None
    child: ArtifactRef | None


class LiveWorker(Document):
    launch: ArtifactRef
    child: ArtifactRef
    stage: Identifier
    synthetic: bool


class Observation(Document):
    specification: ArtifactRef
    sequence: int = Field(ge=1)
    observed_at: str
    cut_monotonic_ns: int = Field(ge=0)
    boot_id: str
    first_check_at: str
    last_check_at: str
    resources: ResourceSample
    live: tuple[LiveWorker, ...]
    attempts: tuple[AttemptSnapshot, ...]
    expected_work: int = Field(ge=0)
    task_states: dict[str, int]
    other_active_plan_digests: tuple[str, ...]


class ResourceSummary(Document):
    samples: int = Field(ge=0)
    timing_bases: tuple[Literal["monotonic", "wall_clock"], ...] = ()
    wall_clock_regressions: int = Field(default=0, ge=0)
    unknown_storage_age_samples: int = Field(default=0, ge=0)
    unmeasured_time_gaps: int = Field(default=0, ge=0)
    maximum_sample_gap_seconds: float | None
    maximum_storage_age_seconds: float | None
    pss_observed_max_bytes: int | None
    rss_observed_max_bytes: int | None
    additional_storage_observed_max_bytes: int | None
    allocated_storage_observed_max_bytes: int | None
    logical_storage_observed_max_bytes: int | None
    entries_observed_max: int | None
    unique_inodes_observed_max: int | None
    minimum_free_observed_bytes: int | None
    owned_vram_observed_max_bytes: int | None
    device_vram_observed_max_bytes: int | None
    unavailable_gpu_samples: int
    gpu_charged_seconds_delta: float | None
    gpu_active_seconds_at_end: float | None
    read_bytes_observed_deltas: int
    write_bytes_observed_deltas: int
    stopped_reasons: tuple[str, ...]
    other_active_plan_digests: tuple[str, ...]


class UsageSummary(Document):
    attempts: int
    reports: int
    missing_reports: int
    known_responses: int
    exact_api_responses: int | None
    missing_response_ids: int
    totals: dict[str, int | None]
    known_subtotals: dict[str, int]
    missing_metric_counts: dict[str, int]
    image_inputs: int
    image_bytes_with_repetition: int
    unique_image_bytes: int
    unbound_image_inputs: int
    prompt_chars: int
    tool_input_chars: int
    tool_output_chars: int
    tool_categories: dict[str, int]
    orphan_outputs: int
    compactions: int
    elapsed_worker_seconds: float


class ConditionReport(Document):
    specification: ArtifactRef
    observations: tuple[ArtifactRef, ...]
    target_workers: Literal[10, 25, 50]
    observed_real_workers_max: int
    observed_synthetic_workers_max: int
    real_workers_by_stage_max: dict[str, int]
    concurrency: Literal["not_observed", "unreached", "synthetic_only", "observed"]
    execution: Literal[
        "not_started", "in_progress", "blocked", "budget_stopped", "workers_complete"
    ]
    task_states: dict[str, int]
    resources: ResourceSummary
    usage: UsageSummary


class ComparisonReport(Document):
    specification: ArtifactRef
    conditions: tuple[ConditionReport, ...]
    quality: Literal["policy_pending", "evidence_pending", "requires_task_evaluation"]
    acceptance: Literal["unverified"] = "unverified"
    limitations: tuple[str, ...]
