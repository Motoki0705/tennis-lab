"""Registered synthetic worker attempts; never launch Codex or invoke a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, Document, FrameRange
from src.tasks.base.annotation.editing.contracts import EditReceipt
from src.tasks.base.annotation.images.contracts import Crop
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import (
    AttemptResult,
    ExitReceipt,
    Launch,
    WorkOutcome,
)
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from src.tasks.base.annotation.workers.review.contracts import (
    EditRequest,
    ImageRequest,
    ImageTile,
    ReviewWorkerResult,
)
from src.tasks.base.annotation.workers.review.service import execute_service
from src.tasks.base.annotation.workers.services import ServiceRequest
from src.tasks.player_detection.annotation.review.contracts import (
    ConfirmPredictions,
    PlayerEdits,
)
from src.tasks.player_detection.annotation.review.worker import (
    PlayerReviewServices,
    PlayerReviewWorker,
)


class ReviewSession:
    def __init__(self, root: Path, **environment_options: Any) -> None:
        from tests.support.tasks.player_detection.annotation import Environment

        self.env = Environment(root, **environment_options)
        self.worker, self.services = PlayerReviewWorker(), PlayerReviewServices()
        self.request = read_document(self.env.layout, self.env.actor, Launch).request
        self.ledger = WorkerLedger(self.env.layout)
        self.ledger.register(
            self.env.plan,
            self.request,
            dependencies=self.worker.input_references(
                self.env.layout, self.env.plan, self.request
            ),
        )
        self.open(1)

    def open(self, number: int) -> None:
        env = self.env
        self.directory = env.layout.path(
            "campaigns",
            env.plan.campaign_id,
            "workers",
            self.request.work_id,
            f"attempt-{number:04d}",
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "work").mkdir(exist_ok=True)
        prepared = self.worker.prepare(
            env.layout, env.plan, self.request, self.directory
        )
        launch = Launch(
            work_id=self.request.work_id,
            attempt=f"{self.request.work_id}-{number:04d}",
            plan=env.plan_ref,
            request=self.request,
            prepared=prepared,
            directory=str(self.directory),
            code_root=str(env.layout.repository),
            codex_binary="codex",
            codex_home=str(env.layout.repository / "codex"),
            timeout_seconds=60.0,
            test_root=str(env.layout.root),
        )
        env.actor = write_document(env.layout, self.directory / "launch.json", launch)
        self.input = annotation_reference(env.layout, self.directory / "request.json")
        with self.ledger.registry.transaction() as db:
            db.execute(
                "INSERT INTO worker_attempts VALUES (?,?,?,'running',?,NULL)",
                (
                    launch.attempt,
                    self.request.work_id,
                    number,
                    env.actor.model_dump_json(),
                ),
            )
            db.execute(
                "UPDATE worker_tasks SET state='running',current_attempt=?,result=NULL,reason=NULL WHERE id=?",
                (launch.attempt, self.request.work_id),
            )

    def request_service(self, operation: str, parameters: Document) -> ArtifactRef:
        service = ServiceRequest(
            task_request=self.input,
            operation=operation,
            parameters=parameters.model_dump(mode="json"),
        )
        result: ArtifactRef = write_document(
            self.env.layout,
            self.directory / "work/service-requests" / (service.sha256 + ".json"),
            service,
        )
        return result

    def call(self, operation: str, parameters: Document) -> ArtifactRef:
        ref = self.request_service(operation, parameters)
        result: ArtifactRef = execute_service(
            self.env.layout, self.env.plan, ref, self.services
        )
        return result

    def image(
        self,
        *frames: int,
        drawing: Literal["plain", "prediction", "revision"] = "plain",
    ) -> RenderedImage:
        ref = self.call(
            "review_image",
            ImageRequest(
                expected_revision=self.env.head(),
                drawing=drawing,
                tiles=tuple(
                    ImageTile(
                        frame_index=i, crop=Crop(left=0, top=0, right=64, bottom=32)
                    )
                    for i in frames
                ),
                columns=min(4, len(frames)),
                tile_width=64,
                tile_height=32,
                reason="synthetic review",
            ),
        )
        result: RenderedImage = read_document(self.env.layout, ref, RenderedImage)
        return result

    def confirm(self, image: RenderedImage, start: int, stop: int) -> EditReceipt:
        edits = PlayerEdits(
            operations=(
                ConfirmPredictions(
                    scope=FrameRange(start=start, stop=stop),
                    evidence=(image.sha256,),
                ),
            ),
            images=(image,),
        )
        changes = write_document(
            self.env.layout, self.directory / "work" / (edits.sha256 + ".json"), edits
        )
        ref = self.call(
            "review_edit",
            EditRequest(
                expected_revision=self.env.head(),
                changes=changes,
                reason="synthetic image confirmation",
            ),
        )
        result: EditReceipt = read_document(self.env.layout, ref, EditReceipt)
        return result

    def evidence(
        self,
        *,
        shown: tuple[RenderedImage, ...] | None = None,
        status: Literal[
            "succeeded", "failed", "quota", "timed_out", "cancelled"
        ] = "succeeded",
        telemetry_fault: Literal["missing_model", "conflicting_response_id"]
        | None = None,
    ) -> ArtifactRef:
        env = self.env
        catalog = self.worker.telemetry_images(
            env.layout, env.plan, self.request, self.directory
        )
        report = env.telemetry(
            catalog, shown=shown, exit_status=status, telemetry_fault=telemetry_fault
        )
        write_document(env.layout, self.directory / "telemetry.ref.json", report)
        return report

    def finish(
        self,
        report: ArtifactRef,
        *,
        status: Literal["complete", "deferred"] = "complete",
        record: bool = True,
    ) -> WorkOutcome:
        env = self.env
        result = write_document(
            env.layout,
            self.directory / "result.json",
            ReviewWorkerResult(
                status=status,
                revision=env.head(),
                reason="synthetic completion"
                if status == "complete"
                else "synthetic handoff",
            ),
        )
        exit_ref = annotation_reference(env.layout, self.directory / "exit.json")
        execution = read_document(env.layout, exit_ref, ExitReceipt)
        outcome = self.worker.validate(
            env.layout,
            env.plan,
            self.request,
            self.directory,
            result,
            execution,
            report,
        )
        if record:
            receipt = write_document(
                env.layout,
                self.directory / "result.receipt.json",
                AttemptResult(
                    launch=env.actor,
                    exit=exit_ref,
                    result=result,
                    outcome=outcome,
                    telemetry=report,
                ),
            )
            self.state(outcome.status, receipt)
        return outcome

    def state(self, state: str, result: ArtifactRef | None = None) -> None:
        with self.ledger.registry.transaction() as db:
            db.execute(
                "UPDATE worker_tasks SET state=?,result=? WHERE id=?",
                (
                    state,
                    result.model_dump_json() if result else None,
                    self.request.work_id,
                ),
            )
            db.execute(
                "UPDATE worker_attempts SET state=? WHERE id=(SELECT current_attempt FROM worker_tasks WHERE id=?)",
                (state, self.request.work_id),
            )
