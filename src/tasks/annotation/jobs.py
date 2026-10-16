"""Resource-accounted immutable operations shared by workflow composition roots."""

from __future__ import annotations

from typing import Any, Literal

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import (
    AcquisitionReceipt,
    SourceCandidate,
)
from src.tasks.base.annotation.collection.ledger import CollectionLedger
from src.tasks.base.annotation.collection.retirement import verify_acquisition_media
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.execution.contracts import StageLaunch
from src.tasks.base.annotation.execution.runner import StageRunner
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    CampaignPlan,
    references,
    require_execution_layout,
)
from src.tasks.base.annotation.predictions import PredictionContext, PredictionReceipt
from src.tasks.base.annotation.resources import (
    CapacityUnavailable,
    ResourceManager,
    ResourceRequest,
)
from src.utils.paths import PROJECT_ROOT

from .contracts import (
    AcquireOperation,
    OperationInput,
    OperationReceipt,
    PredictOperation,
    PrepareOperation,
    Runtime,
    ScreenOperation,
)


class Operations:
    def __init__(
        self,
        layout: AnnotationLayout,
        runtime: Runtime,
        plan: CampaignPlan,
        resources: ResourceManager,
        *,
        command: Literal["stage", "benchmark-stage"] = "stage",
    ) -> None:
        self.layout, self.runtime, self.plan, self.resources = (
            layout,
            runtime,
            plan,
            resources,
        )
        self.command = command
        require_execution_layout(layout, plan)
        if (plan.phase == "benchmark") != (command == "benchmark-stage"):
            raise ValueError("operation command must match the plan phase")
        if (
            read_document(layout, runtime.plan, CampaignPlan) != plan
            or runtime.scope != resources.reference
        ):
            raise ValueError(
                "operation runtime differs from its plan or resource scope"
            )
        self.stages = StageRunner(layout)

    def operation(
        self, data: OperationInput, *, attempt: int | None = None
    ) -> dict[str, Any]:
        if data.plan != self.runtime.plan or (attempt is not None and attempt < 1):
            raise ValueError(
                "operation belongs to another runtime, or has an invalid attempt"
            )
        jobs = self.layout.path("campaigns", self.plan.campaign_id, "jobs")
        attempts = [
            int(path.name.rsplit("-", 1)[1])
            for path in jobs.glob(data.logical_id + "-*")
        ]
        latest = max(attempts, default=0)
        if attempt is None:
            attempt = max(latest, 1)
            if latest:
                prior = read_document(
                    self.layout,
                    annotation_reference(
                        self.layout,
                        jobs / f"{data.logical_id}-{attempt}" / "operation.json",
                    ),
                    OperationInput,
                )
                if prior.logical_id != data.logical_id:
                    raise ValueError("stage attempt differs from its logical operation")
                if (
                    isinstance(data.operation, AcquireOperation)
                    and isinstance(prior.operation, AcquireOperation)
                    and data.operation.retry
                    and not prior.operation.retry
                ):
                    attempt += 1
                else:
                    # Ordinary resume recovers the pinned retry directive as well.
                    data = prior
        if attempt not in attempts:
            if attempt != latest + 1:
                raise ValueError(
                    "a new attempt must follow the latest recorded attempt"
                )
            if latest:
                previous = annotation_reference(
                    layout=self.layout,
                    path=jobs / f"{data.logical_id}-{latest}" / "launch.json",
                )
                previous_state = self.stages.status(previous)
                if previous_state["state"] not in {
                    "failed",
                    "cancelled",
                    "timed_out",
                    "interrupted",
                }:
                    raise ValueError(
                        "only a failed or interrupted operation can be retried"
                    )
        cost = self.runtime.cost(data.stage)
        request_id = f"{data.logical_id}-{attempt}"
        directory = self.layout.path(
            "campaigns", self.plan.campaign_id, "jobs", request_id
        )
        reference = write_document(self.layout, directory / "operation.json", data)
        command = [
            str(PROJECT_ROOT / ".venv/bin/python"),
            "-m",
            "src.tasks.annotation",
            self.command,
            "--operation",
            str(self.layout.artifact_path(reference)),
            "--sha256",
            reference.digest,
        ]
        if self.command == "stage":
            command += ["--attempt", str(attempt)]
        if self.layout.mode == "test":
            command += ["--test-root", str(self.layout.root)]
        launch = StageLaunch(
            scope=self.runtime.scope,
            resource=ResourceRequest(
                request_id=request_id,
                plan=self.runtime.plan,
                stage=data.stage,
                pss_bytes=cost.pss_bytes,
                storage_bytes=cost.storage_bytes,
                gpu_seconds=cost.gpu_seconds,
            ),
            command=tuple(command),
            input=reference,
            dependencies=tuple(references(data)),
            code_root=str(PROJECT_ROOT),
            timeout_seconds=cost.timeout_seconds,
            queue=self.runtime.queue if cost.gpu_seconds else None,
            test_root=str(self.layout.root) if self.layout.mode == "test" else None,
        )
        path = directory / "launch.json"
        if path.exists():
            launch_ref = annotation_reference(self.layout, path)
            existing = read_document(self.layout, launch_ref, StageLaunch)
            if existing != launch:
                raise ValueError("operation resource/command configuration changed")
            state: dict[str, Any] = self.stages.status(launch_ref)
            # Backlog rejection can leave a launch without a resource request.
            # Both unregistered and pending requests can resume the same attempt.
            with self.resources.registry.transaction() as db:
                row = db.execute(
                    "SELECT state FROM resource_requests WHERE id=?",
                    (self.resources.key(launch.resource),),
                ).fetchone()
            unadmitted = row is None or row[0] == "pending"
            if not unadmitted or state["state"] not in {"pending", "interrupted"}:
                return state
        try:
            submitted: dict[str, Any] = self.stages.submit(launch)
            return submitted
        except (CapacityUnavailable, BlockingIOError) as error:
            return {
                "state": "waiting",
                "operation": reference.model_dump(),
                "reason": str(error),
            }

    def receipt(self, state: dict[str, Any]) -> OperationReceipt:
        launch = read_document(
            self.layout, ArtifactRef.model_validate(state["launch"]), StageLaunch
        )
        if state["state"] != "succeeded" or launch.input is None:
            raise ValueError("only a successful stage can provide operation results")
        path = self.layout.artifact_path(launch.input).parent / "result.json"
        receipt: OperationReceipt = read_document(
            self.layout, annotation_reference(self.layout, path), OperationReceipt
        )
        if receipt.operation != launch.input:
            raise ValueError("stage result belongs to a different operation")
        for ref in receipt.results:
            verify_reference(self.layout, ref)
        data = read_document(self.layout, launch.input, OperationInput)
        if data.plan != self.runtime.plan or data.stage != launch.resource.stage:
            raise ValueError("stage result differs from the runtime plan/stage")
        operation = data.operation
        if isinstance(operation, AcquireOperation) or (
            isinstance(operation, (ScreenOperation, PrepareOperation))
            and receipt.status == "not_applicable"
        ):
            source = CollectionLedger(self.layout).source(
                self.plan, operation.source_id
            )
            if source["acquisition"] is None or receipt.results != (
                ArtifactRef.model_validate_json(source["acquisition"]),
            ):
                raise ValueError(
                    "source receipt differs from its registered acquisition"
                )
            acquisition = read_document(
                self.layout, receipt.results[0], AcquisitionReceipt
            )
            candidate = SourceCandidate.model_validate_json(source["candidate"])
            if (
                acquisition.source.canonical_id != operation.source_id
                or acquisition.candidate_digest != candidate.sha256
                or acquisition.configuration != self.plan.preparation
            ):
                raise ValueError("acquisition receipt differs from the frozen source")
            for dependency in references(acquisition):
                if dependency == acquisition.source.media:
                    verify_acquisition_media(
                        self.layout, self.plan, operation.source_id,
                        receipt.results[0], acquisition,
                    )
                else:
                    verify_reference(self.layout, dependency)
            if receipt.status == "not_applicable" and (
                source["state"] != "excluded"
                or not receipt.reason
                or receipt.reason != source["reason"]
            ):
                raise ValueError("source exclusion differs from its registered reason")
        elif receipt.status == "not_applicable":
            raise ValueError("this operation cannot return a source exclusion")
        if isinstance(data.operation, PredictOperation):
            operation = data.operation
            if receipt.status != "complete" or len(receipt.results) != len(
                operation.clips
            ):
                raise ValueError("prediction stage must return one receipt per clip")
            task = next(
                (task for task in self.plan.tasks if task.task == operation.task), None
            )
            if task is None:
                raise ValueError("prediction task is absent from the frozen plan")
            for clip, ref in zip(operation.clips, receipt.results, strict=True):
                manifest = read_document(self.layout, clip, ClipManifest)
                expected = PredictionContext(
                    task=task.task,
                    manifest=clip,
                    clip_id=manifest.clip_id,
                    model=task.model,
                    configuration=task.inference,
                )
                prediction = read_document(self.layout, ref, PredictionReceipt)
                if (
                    prediction.context != expected
                    or prediction.prediction.request_id != expected.request_id
                ):
                    raise ValueError(
                        "prediction receipt differs from the frozen operation"
                    )
                for dependency in (*references(prediction), *references(manifest)):
                    verify_reference(self.layout, dependency)
        return receipt
