"""Aggregate observed evidence with missing measurements kept explicitly missing."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from ..artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import references
from ..resources.contracts import BudgetScope
from ..telemetry.capture import iter_responses, verify_report
from ..telemetry.contracts import Usage
from ..workers.contracts import ChildReceipt, ExitReceipt, Launch
from .capture import directory
from .contracts import (
    AttemptSnapshot,
    ComparisonReport,
    ConditionReport,
    ConditionSpec,
    Observation,
    ResourceSummary,
    UsageSummary,
)
from .validation import comparison, condition, validate_work


def _seconds(value: str) -> float:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("benchmark timestamps require an explicit timezone")
    return stamp.timestamp()


class _Resources:
    def __init__(self) -> None:
        self.count = 0
        self.last_sample_digest: str | None = None
        self.last_time: float | None = None
        self.last_clock: tuple[str, str] | None = None
        self.last_wall_time: float | None = None
        self.clock_bases: set[str] = set()
        self.wall_regressions = 0
        self.unknown_storage_age = 0
        self.unknown_gaps = 0
        self.gap: float | None = None
        self.storage_age: float | None = None
        self.maxima: dict[str, int] = {}
        self.free: int | None = None
        self.gpu_missing = 0
        self.gpu_start: float | None = None
        self.gpu_end: float | None = None
        self.gpu_active: float | None = None
        self.io: dict[tuple[str, int, int], tuple[int, int]] = {}
        self.reads = self.writes = 0
        self.stops: set[str] = set()
        self.foreign: set[str] = set()

    def add(self, observation: Observation) -> None:
        sample = observation.resources
        self.foreign.update(observation.other_active_plan_digests)
        if sample.stopped_reason:
            self.stops.add(sample.stopped_reason)
        if sample.sha256 == self.last_sample_digest:
            return
        self.last_sample_digest = sample.sha256
        wall_stamp = _seconds(sample.measured_at)
        if self.last_wall_time is not None and wall_stamp < self.last_wall_time:
            self.wall_regressions += 1
        self.last_wall_time = wall_stamp
        if sample.measured_monotonic_ns is not None:
            if sample.clock_boot_id != observation.boot_id:
                raise ValueError("resource monotonic clock belongs to another boot")
            clock = (observation.boot_id, "monotonic")
            stamp = sample.measured_monotonic_ns / 1e9
            age = (
                None
                if sample.storage_measured_monotonic_ns is None
                else (
                    sample.measured_monotonic_ns - sample.storage_measured_monotonic_ns
                )
                / 1e9
            )
        else:
            clock = (observation.boot_id, "wall_clock")
            stamp = wall_stamp
            age = stamp - _seconds(sample.storage_measured_at)
        self.clock_bases.add(clock[1])
        if self.last_clock is not None and clock != self.last_clock:
            self.unknown_gaps += 1
            self.last_time = None
        self.last_clock = clock
        if (age is not None and age < 0) or (
            self.last_time is not None and stamp < self.last_time
        ):
            raise ValueError("resource measurement timestamps moved backwards")
        if self.last_time is not None:
            self.gap = max(self.gap or 0.0, stamp - self.last_time)
        if age is None:
            self.unknown_storage_age += 1
        else:
            self.storage_age = max(self.storage_age or 0.0, age)
        self.last_time = stamp
        self.count += 1
        values = {
            "pss_observed_max_bytes": sample.pss_bytes,
            "rss_observed_max_bytes": sample.rss_bytes,
            "additional_storage_observed_max_bytes": sample.storage.additional_bytes,
            "allocated_storage_observed_max_bytes": sample.storage.allocated_bytes,
            "logical_storage_observed_max_bytes": sample.storage.logical_bytes,
            "entries_observed_max": sample.storage.entries,
            "unique_inodes_observed_max": sample.storage.unique_inodes,
            "owned_vram_observed_max_bytes": sample.gpu.owned_vram_bytes,
            "device_vram_observed_max_bytes": sample.gpu.device_vram_bytes,
        }
        for key, value in values.items():
            if value is not None:
                self.maxima[key] = max(self.maxima.get(key, 0), value)
        self.free = (
            sample.storage.free_bytes
            if self.free is None
            else min(self.free, sample.storage.free_bytes)
        )
        self.gpu_missing += sample.gpu.status == "unavailable"
        if self.gpu_start is None:
            self.gpu_start = sample.gpu_charged_seconds
        if self.gpu_end is not None and sample.gpu_charged_seconds < self.gpu_end:
            raise ValueError(
                "cumulative GPU usage decreased within one benchmark scope"
            )
        self.gpu_end, self.gpu_active = (
            sample.gpu_charged_seconds,
            sample.gpu_active_seconds,
        )
        current = {}
        for process in sample.processes:
            identity = (observation.boot_id, process.pid, process.start_ticks)
            counters = (process.read_bytes, process.write_bytes)
            previous = self.io.get(identity)
            if previous is not None:
                if counters[0] < previous[0] or counters[1] < previous[1]:
                    raise ValueError(
                        "I/O counters decreased for the same process identity"
                    )
                self.reads += counters[0] - previous[0]
                self.writes += counters[1] - previous[1]
            current[identity] = counters
        self.io = current

    def result(self) -> ResourceSummary:
        maxima_names = (
            "pss_observed_max_bytes",
            "rss_observed_max_bytes",
            "additional_storage_observed_max_bytes",
            "allocated_storage_observed_max_bytes",
            "logical_storage_observed_max_bytes",
            "entries_observed_max",
            "unique_inodes_observed_max",
            "owned_vram_observed_max_bytes",
            "device_vram_observed_max_bytes",
        )
        return ResourceSummary(
            **{name: self.maxima.get(name) for name in maxima_names},
            samples=self.count,
            timing_bases=tuple(sorted(self.clock_bases)),
            wall_clock_regressions=self.wall_regressions,
            unknown_storage_age_samples=self.unknown_storage_age,
            unmeasured_time_gaps=self.unknown_gaps,
            maximum_sample_gap_seconds=self.gap,
            maximum_storage_age_seconds=self.storage_age,
            minimum_free_observed_bytes=self.free,
            unavailable_gpu_samples=self.gpu_missing,
            gpu_charged_seconds_delta=None
            if self.gpu_start is None or self.gpu_end is None
            else self.gpu_end - self.gpu_start,
            gpu_active_seconds_at_end=self.gpu_active,
            read_bytes_observed_deltas=self.reads,
            write_bytes_observed_deltas=self.writes,
            stopped_reasons=tuple(sorted(self.stops)),
            other_active_plan_digests=tuple(sorted(self.foreign)),
        )


def _usage(
    layout: AnnotationLayout, attempts: tuple[AttemptSnapshot, ...]
) -> UsageSummary:
    totals: Counter[str] = Counter({name: 0 for name in Usage.model_fields})
    missing: Counter[str] = Counter({name: 0 for name in Usage.model_fields})
    categories: Counter[str] = Counter()
    images: dict[str, int] = {}
    reports = responses = missing_ids = image_count = image_bytes = unbound = (
        prompts
    ) = inputs = outputs = orphans = compactions = 0
    unobserved = 0
    elapsed = 0.0
    observed: set[tuple[str, str, str]] = set()
    for attempt in attempts:
        launch = read_document(layout, attempt.launch, Launch)
        if attempt.execution is not None:
            execution = read_document(layout, attempt.execution, ExitReceipt)
            if execution.launch_digest != attempt.launch.digest:
                raise ValueError("benchmark exit belongs to another launch")
            for ref in references(execution):
                verify_reference(layout, ref)
            elapsed += execution.elapsed_seconds
        if attempt.telemetry is None:
            continue
        report = verify_report(layout, attempt.telemetry)
        if (report.launch, report.execution) != (attempt.launch, attempt.execution):
            raise ValueError("benchmark telemetry belongs to another attempt/exit")
        if report.conflicting_usage_records:
            raise ValueError("conflicting response usage cannot be aggregated")
        reports += 1
        unobserved += report.responses == 0
        missing_ids += report.missing_response_ids
        prompts += report.prompt_chars
        unbound += report.unbound_image_inputs
        orphans += report.orphan_output_count
        compactions += report.compactions
        for response in iter_responses(layout, attempt.telemetry):
            identity = (launch.attempt, report.thread_id, response.response_id)
            if identity in observed:
                raise ValueError("duplicate attempt/thread/response evidence")
            observed.add(identity)
            responses += 1
            for name in Usage.model_fields:
                value = (
                    None if response.usage is None else getattr(response.usage, name)
                )
                if value is None:
                    missing[name] += 1
                else:
                    totals[name] += value
            for output in response.tool_outputs:
                categories[output.category] += 1
                inputs += output.input_chars
                outputs += output.text_chars
                for image in output.images:
                    image_count += 1
                    image_bytes += image.image.size_bytes
                    previous = images.setdefault(
                        image.image.digest, image.image.size_bytes
                    )
                    if previous != image.image.size_bytes:
                        raise ValueError("same image hash has inconsistent byte sizes")
    unavailable = len(attempts) - reports
    return UsageSummary(
        attempts=len(attempts),
        reports=reports,
        missing_reports=unavailable,
        unobserved_response_attempts=unobserved,
        known_responses=responses,
        exact_api_responses=None
        if unavailable or unobserved or missing_ids or not attempts
        else responses,
        missing_response_ids=missing_ids,
        totals={
            name: None
            if missing[name] or missing_ids or unavailable or unobserved or not attempts
            else totals[name]
            for name in Usage.model_fields
        },
        known_subtotals=dict(totals),
        missing_metric_counts=dict(missing),
        image_inputs=image_count,
        image_bytes_with_repetition=image_bytes,
        unique_image_bytes=sum(images.values()),
        unbound_image_inputs=unbound,
        prompt_chars=prompts,
        tool_input_chars=inputs,
        tool_output_chars=outputs,
        tool_categories=dict(sorted(categories.items())),
        orphan_outputs=orphans,
        compactions=compactions,
        elapsed_worker_seconds=elapsed,
    )


def _condition_report(
    layout: AnnotationLayout, reference: ArtifactRef, spec: ConditionSpec
) -> ConditionReport:
    _, _, expected_work = condition(layout, reference)
    scope = read_document(layout, spec.scope, BudgetScope)
    paths = sorted((directory(layout, spec) / "observations").glob("*.json"))
    if len(paths) > spec.maximum_observations:
        raise ValueError("stored benchmark observation limit exceeded")
    resource = _Resources()
    refs: list[ArtifactRef] = []
    attempts: dict[ArtifactRef, AttemptSnapshot] = {}
    real_max = synthetic_max = 0
    by_stage: Counter[str] = Counter()
    last: Observation | None = None
    for number, path in enumerate(paths, 1):
        ref = annotation_reference(layout, path)
        value = read_document(layout, ref, Observation)
        if (
            value.specification != reference
            or value.sequence != number
            or path.name != f"{number:06d}.json"
            or value.expected_work != len(expected_work)
            or value.resources.scope_id != scope.scope_id
            or sum(value.task_states.values()) != value.expected_work
        ):
            raise ValueError("benchmark observation sequence/specification changed")
        if (
            last is not None
            and value.boot_id == last.boot_id
            and value.cut_monotonic_ns < last.cut_monotonic_ns
        ):
            raise ValueError("benchmark process observation time moved backwards")
        refs.append(ref)
        resource.add(value)
        live: Counter[str] = Counter()
        synthetic = 0
        launches: set[ArtifactRef] = set()
        attempt_launches = {attempt.launch for attempt in value.attempts}
        for worker in value.live:
            if worker.launch in launches:
                raise ValueError("duplicate live worker evidence")
            launches.add(worker.launch)
            launch = read_document(layout, worker.launch, Launch)
            child = read_document(layout, worker.child, ChildReceipt)
            expected_synthetic = (
                layout.mode == "test"
                or launch.test_command is not None
                or launch.test_root is not None
            )
            if (
                launch.plan != spec.plan
                or launch.resource_scope != spec.scope
                or worker.stage != launch.request.stage
                or child.launch_digest != worker.launch.digest
                or worker.synthetic != expected_synthetic
                or worker.launch not in attempt_launches
                or child.process is None
                or launch.work_id not in expected_work
            ):
                raise ValueError(
                    "live worker provenance differs from the measured condition"
                )
            if worker.synthetic:
                synthetic += 1
            else:
                live[worker.stage] += 1
        real_max = max(real_max, sum(live.values()))
        synthetic_max = max(synthetic_max, synthetic)
        for stage, count in live.items():
            by_stage[stage] = max(by_stage[stage], count)
        for attempt in value.attempts:
            attempt_launch = read_document(layout, attempt.launch, Launch)
            validate_work(
                layout, attempt_launch.request, attempt_launch.work_id, expected_work
            )
            if (
                attempt_launch.plan != spec.plan
                or attempt_launch.resource_scope != spec.scope
            ):
                raise ValueError("observed attempt differs from the frozen cohort")
            previous = attempts.get(attempt.launch)
            if previous is not None and (
                (previous.child is not None and previous.child != attempt.child)
                or (
                    previous.execution is not None
                    and previous.execution != attempt.execution
                )
                or (
                    previous.telemetry is not None
                    and previous.telemetry != attempt.telemetry
                )
            ):
                raise ValueError(
                    "immutable worker evidence changed across observations"
                )
            attempts[attempt.launch] = attempt
        last = value
    resources = resource.result()
    states = {} if last is None else last.task_states
    concurrency = (
        "not_observed"
        if last is None
        else "observed"
        if real_max >= spec.target_workers
        else "synthetic_only"
        if synthetic_max
        else "unreached"
    )
    execution = (
        "budget_stopped"
        if last is not None and last.resources.stopped_reason
        else "not_started"
        if last is None or not attempts
        else "workers_complete"
        if states == {"complete": last.expected_work}
        else "blocked"
        if set(states) - {"unregistered", "ready", "running", "exited", "complete"}
        else "in_progress"
    )
    return ConditionReport(
        specification=reference,
        observations=tuple(refs),
        target_workers=spec.target_workers,
        observed_real_workers_max=real_max,
        observed_synthetic_workers_max=synthetic_max,
        real_workers_by_stage_max=dict(sorted(by_stage.items())),
        concurrency=concurrency,
        execution=execution,
        task_states=states,
        resources=resources,
        usage=_usage(layout, tuple(attempts.values())),
    )


def build_report(layout: AnnotationLayout, reference: ArtifactRef) -> ArtifactRef:
    spec, conditions = comparison(layout, reference)
    report = ComparisonReport(
        specification=reference,
        conditions=tuple(
            _condition_report(layout, ref, entry)
            for ref, entry in zip(spec.conditions, conditions, strict=True)
        ),
        quality="policy_pending"
        if spec.quality_policy is None
        else "evidence_pending"
        if not spec.quality_evidence
        else "requires_task_evaluation",
        limitations=(
            "Concurrency is sampled live CLI children, including API/tool waits; it is not simultaneous API calls or reserved slots.",
            "Resource maxima are observed scope-wide samples, not guaranteed peaks or isolated per-condition attribution; other active plan digests are retained.",
            "I/O includes only positive deltas of identities present in consecutive samples; exited/unobserved work is not estimated.",
            "GPU charge deltas retain the budget ledger's conservative charges; they are not device utilization or complete per-stage GPU timings.",
            "Sampling gaps, stale storage measurements and unavailable GPU telemetry remain explicit. Cached resource samples are counted once.",
            "Worker completion does not certify quality, publication, learning or acceptance. Task-owned fixed-example evaluation is required.",
            "This evidence collector neither replays clips nor launches conditions. Repeated-input execution requires isolated benchmark review state and a shared budget.",
        ),
    )
    return write_document(
        layout,
        layout.path(
            "campaigns",
            spec.benchmark_id,
            "reports",
            "benchmarks",
            "comparisons",
            report.sha256 + ".json",
        ),
        report,
    )
