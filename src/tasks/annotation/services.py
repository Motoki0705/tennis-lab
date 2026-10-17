"""Validate attempt inboxes and dispatch bounded services through accounted stages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.workers.contracts import Launch, WorkRequest
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.tasks.base.annotation.workers.services import ServiceReply, ServiceRequest

from .contracts import (
    OperationInput,
    OperationReceipt,
    PreviewOperation,
    ReviewOperation,
)
from .operations import preview_input
from .screening import ScreeningAdapter


class ServiceBroker:
    def __init__(
        self,
        layout: AnnotationLayout,
        plan: ArtifactRef,
        maximum_services: int,
        operation: Callable[[OperationInput], dict[str, Any]],
        receipt: Callable[[dict[str, Any]], OperationReceipt],
    ) -> None:
        self.layout, self.plan, self.maximum_services = layout, plan, maximum_services
        self.operation, self.receipt = operation, receipt

    def _admit_service(
        self, pool: WorkerPool, attempt: str, service: ServiceRequest
    ) -> None:
        registry = pool.ledger.registry
        registry.initialize(
            "worker-services.v1",
            """
            CREATE TABLE IF NOT EXISTS worker_service_requests (
                attempt TEXT NOT NULL REFERENCES worker_attempts(id),
                digest TEXT NOT NULL, request TEXT NOT NULL,
                PRIMARY KEY (attempt, digest)
            );
            """,
        )
        encoded = service.canonical_bytes().decode()
        with registry.transaction() as db:
            prior = db.execute(
                "SELECT request FROM worker_service_requests WHERE attempt=? AND digest=?",
                (attempt, service.sha256),
            ).fetchone()
            if prior is not None:
                if prior["request"] != encoded:
                    raise ValueError("registered worker service request changed")
                return
            count = db.execute(
                "SELECT count(*) FROM worker_service_requests WHERE attempt=?",
                (attempt,),
            ).fetchone()[0]
            if count >= self.maximum_services:
                raise ValueError(
                    "worker exceeded its frozen image-service request count"
                )
            db.execute(
                "INSERT INTO worker_service_requests VALUES (?,?,?)",
                (attempt, service.sha256, encoded),
            )

    def process(
        self, request: WorkRequest, row: dict[str, Any], pool: WorkerPool
    ) -> None:
        if row["state"] != "running":
            return
        with pool.ledger.registry.transaction() as db:
            raw = db.execute(
                "SELECT launch FROM worker_attempts WHERE id=?",
                (row["current_attempt"],),
            ).fetchone()
        launch = read_document(
            self.layout, ArtifactRef.model_validate_json(raw[0]), Launch
        )
        if launch.plan != self.plan or launch.request != request:
            raise ValueError("service broker belongs to a different plan or worker")
        directory = Path(launch.directory)
        official = annotation_reference(self.layout, directory / "request.json")
        paths = sorted((directory / "work/service-requests").glob("*.json"))
        for path in paths:
            if path.stat().st_size > 32768:
                raise ValueError("worker service request exceeds 32 KiB")
            service_ref = annotation_reference(self.layout, path)
            service = read_document(self.layout, service_ref, ServiceRequest)
            if service.task_request != official or path.stem != service.sha256:
                raise ValueError(
                    "worker service request changed its task/attempt identity"
                )
            reply_path = directory / "services" / service.sha256 / "reply.json"
            if reply_path.exists():
                continue
            result, error = None, None
            try:
                if request.adapter == ScreeningAdapter.name:
                    _service, task, _attempt = preview_input(self.layout, service_ref)
                    if task.prepared_input != request.payload:
                        raise ValueError("service refers to another worker input")
                    op = OperationInput(
                        plan=self.plan,
                        operation=PreviewOperation(service_request=service_ref),
                    )
                elif request.adapter == "ball_review":
                    from src.tasks.base.annotation.workers.review.service import (
                        service_input,
                    )

                    _service, review = service_input(self.layout, service_ref)
                    if review.input.work != request:
                        raise ValueError("review service refers to another worker")
                    op = OperationInput(
                        plan=self.plan,
                        operation=ReviewOperation(service_request=service_ref),
                    )
                else:
                    raise ValueError("worker has no registered parent service")
                self._admit_service(pool, launch.attempt, service)
                state = self.operation(op)
                if state["state"] in {
                    "running",
                    "waiting",
                    "queued",
                    "draining",
                    "pending",
                }:
                    continue
                if state["state"] != "succeeded":
                    raise RuntimeError(
                        f"service stage {state['state']}: {state.get('reason')}"
                    )
                receipt = self.receipt(state)
                if receipt.status != "complete" or len(receipt.results) != 1:
                    raise ValueError("service stage returned no exact result receipt")
                result = receipt.results[0]
            except (ValueError, RuntimeError) as failure:
                error = f"{type(failure).__name__}: {failure}"
            write_document(
                self.layout,
                reply_path,
                ServiceReply(
                    request_digest=service.sha256,
                    status="succeeded" if result is not None else "failed",
                    result=result,
                    reason=error,
                ),
            )
