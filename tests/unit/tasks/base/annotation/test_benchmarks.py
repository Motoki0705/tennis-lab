from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import (
    read_document,
    write_asset,
)
from src.tasks.base.annotation.benchmarks.capture import live_workers, observe
from src.tasks.base.annotation.benchmarks.contracts import (
    ComparisonReport,
    ComparisonSpec,
    LiveWorker,
    Observation,
)
from src.tasks.base.annotation.benchmarks.report import _Resources, _usage, build_report
from src.tasks.base.annotation.benchmarks.validation import comparison, condition
from src.tasks.base.annotation.editing.contracts import ReviewContext
from src.tasks.base.annotation.predictions import PredictionReceipt
from src.tasks.base.annotation.resources import ResourceManager
from src.tasks.base.annotation.telemetry.contracts import (
    ImageInput,
    ResponseEntry,
    ResponsePage,
    ResponseRecord,
    TelemetryReport,
    ToolOutput,
    Usage,
)
from src.tasks.base.annotation.workers.contracts import ChildIdentity, WorkRequest
from src.tasks.base.annotation.workers.processes import ProcessIdentity
from tests.support.annotation_benchmarks import Fixture, sample


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path / "annotation")


def report(fixture: Fixture) -> ComparisonReport:
    result: ComparisonReport = read_document(
        fixture.layout,
        build_report(fixture.layout, fixture.comparison),
        ComparisonReport,
    )
    return result


def identity(pid: int = 123) -> ChildIdentity:
    return ChildIdentity(
        pid=pid, start_ticks=10, group=100, session=100, boot_id="boot"
    )


def test_configured_fifty_reserved_slots_are_not_observed_concurrency(
    fixture: Fixture,
) -> None:
    fixture.observation(condition=2)
    result = report(fixture)
    assert [row.concurrency for row in result.conditions] == [
        "not_observed",
        "not_observed",
        "unreached",
    ]
    assert result.conditions[2].observed_real_workers_max == 0
    assert result.conditions[2].resources.owned_vram_observed_max_bytes is None
    assert result.quality == "policy_pending"
    assert result.acceptance == "unverified"


@pytest.mark.parametrize("change", ["effort", "code", "threads", "input", "scope"])
def test_comparison_rejects_changed_controls(fixture: Fixture, change: str) -> None:
    spec = fixture.conditions[1]
    plan = fixture.plans[1]
    if change == "effort":
        plan = plan.model_copy(update={"agent_effort": "low"})
    elif change == "code":
        plan = plan.model_copy(update={"code_commit": "f" * 40})
    elif change == "threads":
        stage = plan.resources.stages[0].model_copy(update={"threads_per_process": 2})
        plan = plan.model_copy(
            update={"resources": plan.resources.model_copy(update={"stages": (stage,)})}
        )
    elif change == "input":
        spec = spec.model_copy(update={"predictions": (fixture.blob,)})
    else:
        spec = spec.model_copy(
            update={
                "scope": fixture.save(
                    fixture.manager.scope.model_copy(
                        update={"scope_id": "reset-budget"}
                    )
                )
            }
        )
    spec = spec.model_copy(update={"plan": fixture.save(plan)})
    request = fixture.save(
        ComparisonSpec(
            benchmark_id="benchmark",
            conditions=(fixture.specs[0], fixture.save(spec), fixture.specs[2]),
            quality_policy=None,
        )
    )
    with pytest.raises(ValueError):
        comparison(fixture.layout, request)


def test_condition_rejects_mismatched_capacity_and_corrupt_prediction(
    fixture: Fixture,
) -> None:
    changed = fixture.save(
        fixture.conditions[0].model_copy(update={"target_workers": 25})
    )
    with pytest.raises(ValueError, match="capacity"):
        condition(fixture.layout, changed)
    fixture.layout.artifact_path(fixture.prediction).write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed"):
        condition(fixture.layout, fixture.specs[0])


@pytest.mark.parametrize("state", ["gone", "zombie", "reused", "live"])
def test_only_children_alive_on_both_sides_of_cut_count(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    child = identity()
    attempt = fixture.attempt(process=child)
    process = ProcessIdentity(**child.model_dump())
    second = (
        None
        if state == "gone"
        else (replace(process, start_ticks=11), "S")
        if state == "reused"
        else (process, "Z" if state == "zombie" else "S")
    )
    answers = iter(((process, "S"), second))
    monkeypatch.setattr(
        "src.tasks.base.annotation.benchmarks.capture.process_snapshot",
        lambda _pid: next(answers),
    )
    live, cut, before, after = live_workers(fixture.layout, (attempt,))
    assert len(live) == (state == "live")
    assert cut > 0 and before <= after
    if live:
        assert live[0].synthetic


def test_duplicate_child_identity_is_rejected(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = identity()
    first, second = (
        fixture.attempt(process=child),
        fixture.attempt(number=2, process=child),
    )
    monkeypatch.setattr(
        "src.tasks.base.annotation.benchmarks.capture.process_snapshot",
        lambda _pid: (ProcessIdentity(**child.model_dump()), "S"),
    )
    with pytest.raises(ValueError, match="same CLI child"):
        live_workers(fixture.layout, (first, second))


def test_synthetic_live_worker_never_satisfies_actual_acceptance(
    fixture: Fixture,
) -> None:
    attempt = fixture.attempt(process=identity())
    assert attempt.child is not None
    fixture.observation(
        attempts=(attempt,),
        live=(
            LiveWorker(
                launch=attempt.launch,
                child=attempt.child,
                stage="review",
                synthetic=True,
            ),
        ),
    )
    row = report(fixture).conditions[0]
    assert (
        row.observed_real_workers_max,
        row.observed_synthetic_workers_max,
        row.concurrency,
    ) == (0, 1, "synthetic_only")
    assert row.usage.missing_reports == 1
    assert row.usage.exact_api_responses is None
    assert row.usage.totals["input_tokens"] is None


def test_synthetic_evidence_cannot_be_relabeled_as_real(fixture: Fixture) -> None:
    attempt = fixture.attempt(process=identity())
    assert attempt.child is not None
    fixture.observation(
        attempts=(attempt,),
        live=(
            LiveWorker(
                launch=attempt.launch,
                child=attempt.child,
                stage="review",
                synthetic=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="provenance"):
        report(fixture)


def test_observer_preserves_shared_baseline_and_stop(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = fixture.manager.scope.baseline
    monkeypatch.setattr(
        ResourceManager, "poll", lambda self: sample(stopped="disk reserve")
    )
    with fixture.manager.registry.transaction() as db:
        db.execute("UPDATE resource_scopes SET stop_reason='disk reserve'")
    for ref in fixture.specs:
        observe(fixture.layout, ref)
    result = report(fixture)
    assert all(row.execution == "budget_stopped" for row in result.conditions)
    reopened = ResourceManager(fixture.layout, fixture.manager.reference)
    assert reopened.scope.baseline == baseline
    assert reopened.scope.previously_allocated_bytes == 17
    assert reopened.scope.previously_used_gpu_seconds == 7
    with reopened.registry.transaction() as db:
        assert (
            db.execute("SELECT stop_reason FROM resource_scopes").fetchone()[0]
            == "disk reserve"
        )


def test_observer_records_running_attempt_and_retains_sequence(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.attempt()
    monkeypatch.setattr(ResourceManager, "poll", lambda self: sample())
    one = read_document(
        fixture.layout, observe(fixture.layout, fixture.specs[0]), Observation
    )
    two = read_document(
        fixture.layout, observe(fixture.layout, fixture.specs[0]), Observation
    )
    assert (one.sequence, two.sequence) == (1, 2)
    assert one.task_states == {"running": 1}
    assert len(one.attempts) == 1 and one.live == ()
    assert report(fixture).conditions[0].resources.samples == 1


@pytest.mark.parametrize("boundary", ["observe", "report"])
@pytest.mark.parametrize(
    "change",
    ["receipt", "manifest", "missing_manifest", "payload_receipt", "payload_manifest"],
)
def test_frozen_worker_inputs_reject_same_clip_with_different_references(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, boundary: str, change: str
) -> None:
    receipt = read_document(fixture.layout, fixture.prediction, PredictionReceipt)
    other_manifest = fixture.save(fixture.manifest.model_copy(update={"width": 3}))
    changed_manifest = read_document(
        fixture.layout, other_manifest, type(fixture.manifest)
    )
    assert changed_manifest.clip_id == fixture.manifest.clip_id
    other_prediction = fixture.save(
        receipt.model_copy(
            update={
                "prediction": receipt.prediction.model_copy(
                    update={"artifact": other_manifest}
                )
            }
        )
    )
    prediction = other_prediction if change == "receipt" else fixture.prediction
    manifest = other_manifest if change == "manifest" else receipt.context.manifest
    request = WorkRequest(
        adapter="fixture",
        adapter_version="v1",
        stage="review",
        subject_id=fixture.manifest.clip_id,
        payload=fixture.save(
            ReviewContext(
                plan=fixture.conditions[0].plan,
                task="fixture",
                manifest=other_manifest if change == "payload_manifest" else manifest,
                prediction=other_prediction
                if change == "payload_receipt"
                else prediction,
            )
        ),
        inputs=(prediction,)
        if change == "missing_manifest"
        else (manifest, prediction),
    )
    attempt = fixture.attempt(request=request)
    monkeypatch.setattr(ResourceManager, "poll", lambda self: sample())
    if boundary == "report":
        fixture.observation(attempts=(attempt,))
    with pytest.raises(ValueError, match="frozen"):
        if boundary == "observe":
            observe(fixture.layout, fixture.specs[0])
        else:
            report(fixture)


def test_observation_limit_and_spec_change_are_not_silent_resets(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ResourceManager, "poll", lambda self: sample())
    small = fixture.save(
        fixture.conditions[0].model_copy(update={"maximum_observations": 1})
    )
    observe(fixture.layout, small)
    with pytest.raises(ValueError, match="limit"):
        observe(fixture.layout, small)
    with pytest.raises(ValueError, match="specification changed"):
        observe(fixture.layout, fixture.specs[0])


def test_resource_summary_separates_process_identity_and_stale_samples(
    fixture: Fixture,
) -> None:
    one = read_document(fixture.layout, fixture.observation(), Observation)
    aggregate = _Resources()
    aggregate.add(one)
    aggregate.add(one)  # poll cache is not an additional resource measurement
    aggregate.add(one.model_copy(update={"resources": sample(3)}))
    aggregate.add(one.model_copy(update={"resources": sample(4, ticks=200)}))
    result = aggregate.result()
    assert result.samples == 3
    assert (
        result.pss_observed_max_bytes == 40 and result.maximum_sample_gap_seconds == 2
    )
    assert result.maximum_storage_age_seconds == 4
    assert (result.read_bytes_observed_deltas, result.write_bytes_observed_deltas) == (
        200,
        400,
    )
    assert result.gpu_charged_seconds_delta == 3
    assert (
        result.unavailable_gpu_samples == 3
        and result.device_vram_observed_max_bytes is None
    )


def test_resource_counter_reset_and_foreign_scope_are_rejected(
    fixture: Fixture,
) -> None:
    one = read_document(fixture.layout, fixture.observation(), Observation)
    aggregate = _Resources()
    aggregate.add(one)
    with pytest.raises(ValueError, match="GPU usage decreased"):
        aggregate.add(
            one.model_copy(
                update={
                    "resources": sample(2).model_copy(
                        update={"gpu_charged_seconds": 0.0}
                    )
                }
            )
        )
    fixture.observation(
        number=2, resources=sample(2).model_copy(update={"scope_id": "another"})
    )
    with pytest.raises(ValueError, match="sequence/specification"):
        report(fixture)


def test_unresolved_worker_and_quality_attachment_are_not_passes(
    fixture: Fixture,
) -> None:
    attempt = fixture.attempt(state="deferred")
    fixture.observation(attempts=(attempt,), task_states={"deferred": 1})
    spec = ComparisonSpec(
        benchmark_id="benchmark",
        conditions=fixture.specs,
        quality_policy=fixture.blob,
        quality_evidence=(fixture.blob,),
    )
    result = read_document(
        fixture.layout,
        build_report(fixture.layout, fixture.save(spec)),
        ComparisonReport,
    )
    assert result.conditions[0].execution == "blocked"
    assert (
        result.quality == "requires_task_evaluation"
        and result.acceptance == "unverified"
    )


def test_cli_report_is_immutable_and_schema_is_available(fixture: Fixture) -> None:
    command = [
        sys.executable,
        "-m",
        "src.tasks.annotation",
        "benchmark-report",
        "--operation",
        str(fixture.layout.artifact_path(fixture.comparison)),
        "--sha256",
        fixture.comparison.digest,
        "--test-root",
        str(fixture.layout.root),
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout) == json.loads(second.stdout)
    schema = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.annotation",
            "schema",
            "--type",
            "benchmark_condition",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(schema.stdout)["title"] == "ConditionSpec"


def test_usage_aggregates_unique_responses_images_and_missing_fields(
    fixture: Fixture,
) -> None:
    attempt = fixture.exit(fixture.attempt())
    assert attempt.execution is not None
    picture = write_asset(
        fixture.layout,
        fixture.layout.path("cache", "fixture.png"),
        b"immutable synthetic image bytes",
    )
    image = ImageInput(
        image=picture,
        width=2,
        height=2,
        detail=None,
        rollout_line=1,
        block_index=0,
        source=None,
        binding="unbound",
    )
    output = ToolOutput(
        call_id="tool-1",
        name="render",
        input_digest=fixture.blob.digest,
        input_chars=5,
        input_timestamp=None,
        output_digest=fixture.blob.digest,
        text_chars=7,
        output_line=1,
        timestamp=None,
        images=(image, image),
        observed_view_paths=(),
        category="exploration.v1",
    )
    response = fixture.save(
        ResponseRecord(
            index=1,
            response_id="response-1",
            thread_id="fixture-thread",
            session_id=None,
            turn_id=None,
            root_turn_id=None,
            rollout_line=1,
            timestamp=None,
            usage=Usage(input_tokens=100, output_tokens=10, total_tokens=110),
            logged_thread_usage=None,
            logged_turn_usage=None,
            context_window=None,
            context_proxy_tokens=110,
            tool_outputs=(output,),
        )
    )
    page = fixture.save(
        ResponsePage(
            entries=(
                ResponseEntry(
                    index=1,
                    response_id="response-1",
                    record=response,
                    page=fixture.blob,
                    images=2,
                    text_chars=7,
                    input_tokens=100,
                    output_tokens=10,
                ),
            )
        )
    )
    orphan = write_asset(
        fixture.layout,
        fixture.layout.path("campaigns", "inputs", "orphans.json"),
        b"[]",
    )
    telemetry = fixture.save(
        TelemetryReport(
            launch=attempt.launch,
            execution=attempt.execution,
            thread_id="fixture-thread",
            rollout_path="synthetic.jsonl",
            rollout_digest=fixture.blob.digest,
            rollout_bytes=0,
            cli_version="fixture",
            model_provider="fixture",
            model_contexts=("gpt-6.1-sol",),
            responses=1,
            duplicate_usage_records=9,
            conflicting_usage_records=0,
            missing_response_ids=0,
            missing_usage=0,
            token_notifications=20,
            compactions=1,
            orphan_outputs=orphan,
            orphan_output_count=0,
            pages=(page,),
            index_page=fixture.blob,
            totals={},
            known_subtotals={},
            missing_metric_counts={},
            image_inputs=2,
            unbound_image_inputs=2,
            catalog_digest=fixture.blob.digest,
            classifier_id="fixture.v1",
            prompt_chars=13,
            elapsed_seconds=2.0,
        )
    )
    measured = attempt.model_copy(update={"telemetry": telemetry})
    summary = _usage(fixture.layout, (measured,))
    assert (
        summary.exact_api_responses == 1
    )  # neither notifications nor duplicate usage records
    assert summary.totals["input_tokens"] == 100
    assert summary.totals["cached_input_tokens"] is None
    assert summary.image_bytes_with_repetition == picture.size_bytes * 2
    assert summary.unique_image_bytes == picture.size_bytes
    assert summary.tool_categories == {"exploration.v1": 1}
    assert (
        summary.tool_input_chars,
        summary.tool_output_chars,
        summary.compactions,
    ) == (5, 7, 1)
    missing = _usage(fixture.layout, (measured, fixture.attempt(number=2)))
    assert (
        missing.exact_api_responses is None and missing.totals["input_tokens"] is None
    )
    assert missing.known_subtotals["input_tokens"] == 100


def test_monotonic_resource_timing_survives_wall_clock_regression(
    fixture: Fixture,
) -> None:
    first = sample(1).model_copy(
        update={
            "clock_boot_id": "boot",
            "measured_monotonic_ns": 5_000_000_000,
            "storage_measured_monotonic_ns": 3_000_000_000,
        }
    )
    second = sample(2).model_copy(
        update={
            "measured_at": "2026-10-14T23:59:58+00:00",
            "clock_boot_id": "boot",
            "measured_monotonic_ns": 8_000_000_000,
            "storage_measured_monotonic_ns": 3_000_000_000,
        }
    )
    fixture.observation(resources=first)
    fixture.observation(number=2, resources=second)
    result = report(fixture).conditions[0].resources
    assert result.maximum_sample_gap_seconds == 3
    assert result.maximum_storage_age_seconds == 5
    assert result.wall_clock_regressions == 1
    assert result.timing_bases == ("monotonic",)
    assert result.unmeasured_time_gaps == result.unknown_storage_age_samples == 0


def test_monotonic_resource_clock_must_match_boot_and_cannot_regress(
    fixture: Fixture,
) -> None:
    observation = read_document(fixture.layout, fixture.observation(), Observation)
    base = sample(1).model_copy(
        update={
            "clock_boot_id": "boot",
            "measured_monotonic_ns": 5_000_000_000,
            "storage_measured_monotonic_ns": 3_000_000_000,
        }
    )
    aggregate = _Resources()
    aggregate.add(observation.model_copy(update={"resources": base}))
    with pytest.raises(ValueError, match="another boot"):
        aggregate.add(
            observation.model_copy(
                update={"resources": base.model_copy(update={"clock_boot_id": "other"})}
            )
        )
    with pytest.raises(ValueError, match="moved backwards"):
        aggregate.add(
            observation.model_copy(
                update={
                    "resources": base.model_copy(
                        update={"measured_monotonic_ns": 4_000_000_000}
                    )
                }
            )
        )


def test_mixed_legacy_clock_and_missing_storage_age_are_explicit(
    fixture: Fixture,
) -> None:
    one = read_document(fixture.layout, fixture.observation(), Observation)
    aggregate = _Resources()
    aggregate.add(one)
    aggregate.add(
        one.model_copy(
            update={
                "resources": sample(2).model_copy(
                    update={
                        "clock_boot_id": "boot",
                        "measured_monotonic_ns": 8_000_000_000,
                    }
                )
            }
        )
    )
    result = aggregate.result()
    assert result.timing_bases == ("monotonic", "wall_clock")
    assert result.unknown_storage_age_samples == result.unmeasured_time_gaps == 1
    assert result.maximum_sample_gap_seconds is None
