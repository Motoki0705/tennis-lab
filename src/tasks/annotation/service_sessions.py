"""Bounded CPU sessions serve live campaign inboxes under one supervised lease."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from functools import partial
from typing import Any

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, Document
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan
from src.tasks.base.annotation.workers.contracts import WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from src.tasks.base.annotation.workers.services import ServiceReply

from .contracts import (
    OperationInput,
    OperationReceipt,
    Runtime,
    ServiceSessionOperation,
)
from .services import ServiceBroker


class MemberReceipt(Document):
    session: ArtifactRef
    worker: ArtifactRef
    operation: ArtifactRef
    result: ArtifactRef | None
    reply: ArtifactRef


class SessionReceipt(Document):
    operation: ArtifactRef
    members: tuple[ArtifactRef, ...]
    elapsed_seconds: float
    reason: str


def execute_session(layout: AnnotationLayout, reference: ArtifactRef) -> ArtifactRef:
    """The session owns all child operations; every input/result/reply stays immutable."""
    from .operations import _execute
    from .service_completion import execute_and_reply

    data = read_document(layout, reference, OperationInput)
    op = data.operation
    if (
        not isinstance(op, ServiceSessionOperation)
        or os.environ.get("TENNIS_ANNOTATION_STAGE_OPERATION") != reference.digest
    ):
        raise ValueError("service session requires its resource-accounted stage")
    plan = read_document(layout, data.plan, CampaignPlan)
    if plan.phase == "benchmark":
        raise ValueError("benchmark sessions require the dedicated benchmark runner")
    directory = layout.artifact_path(reference).parent
    ledger = WorkerLedger(layout)

    def forbidden(_data: Any) -> Any:
        raise RuntimeError("session members cannot acquire another resource lease")

    broker = ServiceBroker(
        layout, data.plan, op.maximum_services_per_attempt, forbidden, forbidden
    )
    members: list[ArtifactRef] = []
    started = last_work = time.monotonic()
    reason = "duration limit"
    while time.monotonic() - started < op.duration_seconds:
        with ledger.registry.transaction() as db:
            budget = db.execute(
                "SELECT s.stop_reason FROM resource_campaigns c "
                "JOIN resource_scopes s ON s.id=c.scope "
                "WHERE c.campaign=? AND c.plan=?",
                (plan.campaign_id, plan.sha256),
            ).fetchone()
            if budget is None:
                raise ValueError(
                    "service session requires its registered campaign budget"
                )
            if budget[0] is not None:
                reason = f"budget stopped: {budget[0]}"
                break
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM worker_tasks WHERE campaign=? AND state='running'",
                    (plan.campaign_id,),
                )
            ]
        for row in rows:
            if row["plan_digest"] != plan.sha256:
                raise ValueError("service session encountered a different worker plan")
            request = WorkRequest.model_validate_json(row["request"])
            broker.collect(request, row, ledger)
        item = broker.next()
        if item is None:
            if not rows or time.monotonic() - last_work >= min(30, op.duration_seconds):
                reason = "no live workers" if not rows else "idle"
                break
            time.sleep(0.1)
            continue
        member_directory = directory / "services" / item.service.sha256
        member_input = write_document(
            layout, member_directory / "operation.json", item.operation
        )
        result = None
        failure = None
        try:
            result = execute_and_reply(
                layout,
                member_input,
                partial(_execute, layout, member_input, session=reference),
            )
        except (OSError, ValueError, RuntimeError) as error:
            # A failed request has an immutable failure reply. The worker may defer;
            # neither its input nor a stale revision is retried here.
            failure = f"{type(error).__name__}: {error}"
        reply_ref = annotation_reference(
            layout,
            layout.artifact_path(item.service.task_request).parent
            / "services"
            / item.service.sha256
            / "reply.json",
        )
        reply = read_document(layout, reply_ref, ServiceReply)
        if (
            reply.request_digest != item.service.sha256
            or (reply.status == "succeeded") != (result is not None)
            or (failure is not None and reply.reason != failure)
        ):
            raise ValueError("session member result and reply disagree")
        members.append(
            write_document(
                layout,
                member_directory / "receipt.json",
                MemberReceipt(
                    session=reference,
                    worker=annotation_reference(
                        layout,
                        layout.artifact_path(item.service.task_request).parent
                        / "launch.json",
                    ),
                    operation=member_input,
                    result=result,
                    reply=reply_ref,
                ),
            )
        )
        last_work = time.monotonic()
        if len(members) >= op.maximum_requests:
            reason = "request limit"
            break
    return write_document(
        layout,
        directory / "session.json",
        SessionReceipt(
            operation=reference,
            members=tuple(members),
            elapsed_seconds=time.monotonic() - started,
            reason=reason,
        ),
    )


class ServiceSessions:
    def __init__(
        self,
        layout: AnnotationLayout,
        runtime: Runtime,
        campaign: str,
        operation: Callable[[OperationInput], dict[str, Any]],
        receipt: Callable[[dict[str, Any]], OperationReceipt],
    ) -> None:
        self.layout, self.runtime, self.operation, self.receipt = (
            layout,
            runtime,
            operation,
            receipt,
        )
        self.directory = layout.path("campaigns", campaign, "service-sessions")
        paths = sorted(self.directory.glob("session-*.json"))
        self.current = (
            read_document(
                layout, annotation_reference(layout, paths[-1]), OperationInput
            )
            if paths
            else None
        )
        self.number = 0
        if self.current is not None:
            if self.current.plan != runtime.plan or not isinstance(
                self.current.operation, ServiceSessionOperation
            ):
                raise ValueError("service session index belongs to another campaign")
            self.number = self.current.operation.session

    def tick(self, broker: ServiceBroker) -> None:
        pending = bool(broker.pending)
        broker.pending.clear()
        if self.current is not None:
            state = self.operation(self.current)
            if state["state"] in {
                "running",
                "waiting",
                "pending",
                "queued",
                "draining",
            }:
                return
            if state["state"] != "succeeded":
                # A failed session remains visible until an explicit stage retry.
                return
            receipt = self.receipt(state)
            if receipt.status != "complete" or len(receipt.results) != 1:
                raise ValueError("service session lacks a complete receipt")
            report = read_document(self.layout, receipt.results[0], SessionReceipt)
            if report.operation != receipt.operation:
                raise ValueError("service session report belongs to another operation")
            for ref in report.members:
                member = read_document(self.layout, ref, MemberReceipt)
                if member.session != report.operation:
                    raise ValueError("service session member has another owner")
                for dependency in (
                    member.worker,
                    member.operation,
                    member.reply,
                    member.result,
                ):
                    if dependency is not None:
                        verify_reference(self.layout, dependency)
            self.current = None
        if not pending:
            return
        self.number += 1
        self.current = OperationInput(
            plan=self.runtime.plan,
            operation=ServiceSessionOperation(
                session=self.number,
                maximum_requests=128,
                maximum_services_per_attempt=self.runtime.maximum_services_per_attempt,
                duration_seconds=min(
                    600, self.runtime.cost("render").timeout_seconds / 2
                ),
            ),
        )
        write_document(
            self.layout,
            self.directory / f"session-{self.number:08d}.json",
            self.current,
        )
        self.operation(self.current)
