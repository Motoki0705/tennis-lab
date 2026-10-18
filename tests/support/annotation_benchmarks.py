"""Tiny immutable evidence, no video decoding, model inference, API or GPU use."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.benchmarks.capture import directory
from src.tasks.base.annotation.benchmarks.contracts import (
    AttemptSnapshot,
    ComparisonSpec,
    ConditionSpec,
    Observation,
    WorkerBinding,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    Document,
    FrameMap,
    FrameRange,
    ModelRef,
    PredictionRef,
    SourceIdentity,
)
from src.tasks.base.annotation.editing.contracts import ReviewContext
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, save_plan
from src.tasks.base.annotation.predictions import PredictionContext, PredictionReceipt
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.resources.contracts import (
    GPUUsage,
    ProcessUsage,
    ResourceSample,
    StorageUsage,
)
from src.tasks.base.annotation.workers.contracts import (
    ChildIdentity,
    ChildReceipt,
    ExitReceipt,
    Launch,
    PreparedWorker,
    WorkRequest,
)
from src.tasks.base.annotation.workers.ledger import WorkerLedger


class Fixture:
    def __init__(self, root: Path) -> None:
        self.layout = AnnotationLayout.testing(root)
        self.blob = write_asset(
            self.layout,
            self.layout.path("campaigns", "inputs", "fixture.txt"),
            b"synthetic only",
        )
        self.manifest = ClipManifest(
            source=SourceIdentity(
                provider="local",
                channel_id="fixture",
                video_id="fixture",
                media=self.blob,
                origin_url="local:fixture",
            ),
            media=self.blob,
            preparation_digest=self.blob.digest,
            width=2,
            height=2,
            time_base="1/30",
            nominal_fps="30",
            source_range=FrameRange(start=0, stop=1),
            target_range=FrameRange(start=0, stop=1),
            frames=(
                FrameMap(
                    frame_index=0,
                    source_frame_index=0,
                    source_pts=0,
                    clip_pts=0,
                    duration_pts=1,
                ),
            ),
        )
        model = ModelRef(
            task="fixture",
            version="v1",
            training_weights=self.blob,
            inference_weights=self.blob,
            configuration=self.blob,
        )
        context = PredictionContext(
            task="fixture",
            manifest=self.save(self.manifest),
            clip_id=self.manifest.clip_id,
            model=model,
            configuration=self.blob,
        )
        self.prediction = self.save(
            PredictionReceipt(
                context=context,
                prediction=PredictionRef(
                    task=context.task,
                    clip_id=context.clip_id,
                    manifest_digest=context.manifest.digest,
                    model_digest=model.sha256,
                    inference_digest=self.blob.digest,
                    artifact=self.blob,
                ),
            )
        )
        ref = self.blob.model_dump(mode="json")
        raw: dict[str, Any] = {
            "phase": "preview",
            "annotation_root": str(root),
            "counts": {
                "channels": 1,
                "videos_per_channel": 1,
                "max_clips_per_video": 1,
            },
            "selection": ref,
            "preparation": ref,
            "code_commit": "a" * 40,
            "code_diff_digest": "b" * 64,
            "agent_model": "gpt-6.1-sol",
            "agent_effort": "max",
            "resources": {
                "additional_storage_bytes": 100_000_000,
                "minimum_free_bytes": 1,
                "total_pss_bytes": 1024**3,
                "gpu_seconds": 60,
            },
            "tasks": [
                {
                    "task": "fixture",
                    "model": model.model_dump(mode="json"),
                    "inference": ref,
                    "review": None,
                    "learning": None,
                    "products": [
                        {
                            "product": "fixture",
                            "dataset_task": "fixture",
                            "dataset_version": "v1",
                            "component": "frames",
                            "base_snapshot": None,
                            "generation_policy": ref,
                        }
                    ],
                }
            ],
        }
        self.plans: list[CampaignPlan] = []
        self.conditions: list[ConditionSpec] = []
        for count in (10, 25, 50):
            raw["campaign_id"] = f"condition-{count}"
            raw["resources"]["codex_workers"] = count
            raw["resources"]["stages"] = [
                {
                    "stage": "review",
                    "kind": "codex",
                    "parallel": count,
                    "threads_per_process": 1,
                }
            ]
            self.plans.append(CampaignPlan.model_validate_json(json.dumps(raw)))
        self.manager = ResourceManager.create(
            self.layout,
            scope_id="shared-budget",
            limits=BudgetLimits.from_budget(self.plans[0].resources),
            roots=(self.layout.repository,),
            previously_allocated_bytes=17,
            previously_used_gpu_seconds=7.0,
            max_pending_per_stage=50,
        )
        for plan in self.plans:
            self.conditions.append(
                ConditionSpec(
                    benchmark_id="benchmark",
                    condition_id=plan.campaign_id,
                    plan=annotation_reference(
                        self.layout, save_plan(self.layout, plan)
                    ),
                    scope=self.manager.reference,
                    predictions=(self.prediction,),
                    bindings=(
                        WorkerBinding(
                            stage="review",
                            adapter="fixture",
                            adapter_version="v1",
                            task="fixture",
                        ),
                    ),
                    target_workers=plan.resources.codex_workers,
                    maximum_observations=10,
                )
            )
        self.specs = tuple(self.save(spec) for spec in self.conditions)
        self.comparison = self.save(
            ComparisonSpec(
                benchmark_id="benchmark", conditions=self.specs, quality_policy=None
            )
        )

    def save(self, document: Document) -> ArtifactRef:
        result: ArtifactRef = write_document(
            self.layout,
            self.layout.path("campaigns", "inputs", document.sha256 + ".json"),
            document,
        )
        return result

    def attempt(
        self,
        *,
        number: int = 1,
        condition: int = 0,
        state: str = "running",
        process: ChildIdentity | None = None,
        request: WorkRequest | None = None,
    ) -> AttemptSnapshot:
        spec, plan = self.conditions[condition], self.plans[condition]
        request = request or WorkRequest(
            adapter="fixture",
            adapter_version="v1",
            stage="review",
            subject_id=self.manifest.clip_id,
            payload=self.save(
                ReviewContext(
                    plan=spec.plan,
                    task="fixture",
                    manifest=self.save(self.manifest),
                    prediction=self.prediction,
                )
            ),
            inputs=(self.save(self.manifest), self.prediction),
        )
        ledger = WorkerLedger(self.layout)
        ledger.register(plan, request, dependencies=())
        path = self.layout.path(
            "campaigns",
            plan.campaign_id,
            "workers",
            request.work_id,
            f"attempt-{number:04d}",
        )
        launch = Launch(
            work_id=request.work_id,
            attempt=f"attempt-{number:04d}",
            plan=spec.plan,
            request=request,
            prepared=PreparedWorker(prompt=self.blob),
            directory=str(path),
            code_root=str(self.layout.repository),
            codex_binary="synthetic-cli",
            codex_home=str(self.layout.root / "codex"),
            timeout_seconds=30,
            test_command=("synthetic",),
            test_root=str(self.layout.root),
            resource_scope=spec.scope,
            resource_request=ResourceRequest(
                request_id=f"worker-{number}",
                plan=spec.plan,
                stage="review",
                pss_bytes=1024,
                storage_bytes=0,
                gpu_seconds=0,
            ),
        )
        launch_ref = write_document(self.layout, path / "launch.json", launch)
        child_ref = write_document(
            self.layout,
            path / "child.json",
            ChildReceipt(
                launch_digest=launch_ref.digest,
                captured_at="2026-10-15T00:00:00+00:00",
                process=process,
                unavailable_reason="fixture does not start a process"
                if process is None
                else None,
            ),
        )
        with ledger.registry.transaction() as db:
            db.execute(
                "INSERT INTO worker_attempts VALUES (?,?,?,?,?,NULL)",
                (
                    launch.attempt,
                    request.work_id,
                    number,
                    state,
                    launch_ref.canonical_bytes().decode(),
                ),
            )
            db.execute(
                "UPDATE worker_tasks SET state=?,current_attempt=? WHERE id=?",
                (state, launch.attempt, request.work_id),
            )
        return AttemptSnapshot(
            launch=launch_ref,
            state=state,
            execution=None,
            telemetry=None,
            child=child_ref,
        )

    def exit(self, attempt: AttemptSnapshot) -> AttemptSnapshot:
        from src.tasks.base.annotation.artifacts import read_document

        launch = read_document(self.layout, attempt.launch, Launch)
        ref = write_document(
            self.layout,
            Path(launch.directory) / "exit.json",
            ExitReceipt(
                launch_digest=attempt.launch.digest,
                exit_code=0,
                status="succeeded",
                started_at="2026-10-15T00:00:00+00:00",
                ended_at="2026-10-15T00:00:02+00:00",
                elapsed_seconds=2.0,
                events=self.blob,
                errors=self.blob,
                thread_id="fixture-thread",
                reason=None,
            ),
        )
        result: AttemptSnapshot = attempt.model_copy(
            update={"execution": ref, "state": "complete"}
        )
        return result

    def observation(
        self,
        *,
        condition: int = 0,
        number: int = 1,
        attempts: tuple[AttemptSnapshot, ...] = (),
        **changes: Any,
    ) -> ArtifactRef:
        spec = self.conditions[condition]
        values: dict[str, Any] = dict(
            specification=self.specs[condition],
            sequence=number,
            observed_at=f"2026-10-15T00:00:{number:02d}+00:00",
            cut_monotonic_ns=number,
            boot_id="boot",
            first_check_at="2026-10-15T00:00:00+00:00",
            last_check_at="2026-10-15T00:00:01+00:00",
            resources=sample(number),
            live=(),
            attempts=attempts,
            expected_work=1,
            task_states={"unregistered": 1} if not attempts else {"running": 1},
            other_active_plan_digests=(),
        )
        values.update(changes)
        return write_document(
            self.layout,
            directory(self.layout, spec) / "observations" / f"{number:06d}.json",
            Observation(**values),
        )


def sample(
    number: int = 1, *, stopped: str | None = None, pid: int = 1, ticks: int = 100
) -> ResourceSample:
    stamp = f"2026-10-15T00:00:{number:02d}+00:00"
    return ResourceSample(
        measured_at=stamp,
        storage_measured_at="2026-10-15T00:00:00+00:00",
        scope_id="shared-budget",
        storage=StorageUsage(
            allocated_bytes=100 + number,
            additional_bytes=17 + number,
            logical_bytes=200,
            entries=10,
            unique_inodes=9,
            free_bytes=1000 - number,
        ),
        processes=(
            ProcessUsage(
                pid=pid,
                start_ticks=ticks,
                group=1,
                session=1,
                pss_bytes=10 * number,
                rss_bytes=20 * number,
                read_bytes=100 * number,
                write_bytes=200 * number,
            ),
        ),
        gpu=GPUUsage(
            status="unavailable",
            owned_vram_bytes=None,
            device_vram_bytes=None,
            maximum_utilization_percent=None,
            reason="synthetic: no GPU query",
        ),
        pss_bytes=10 * number,
        rss_bytes=20 * number,
        gpu_charged_seconds=7.0 + number,
        gpu_active_seconds=0,
        gpu_reserved_seconds=0,
        active_by_stage={"review": 50},
        pending_by_stage={},
        stopped_reason=stopped,
    )
