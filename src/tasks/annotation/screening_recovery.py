"""Withdraw only an unprepared selection sealed with unresolved image requests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
)
from src.tasks.base.annotation.collection.contracts import AcquisitionReceipt
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.workers.contracts import AttemptResult, WorkRequest

from .contracts import RecoveryInput, ReopenScreening
from .screening import unfinished_services

if TYPE_CHECKING:
    from .controller import Controller


def reopen_incomplete_screening(
    controller: Controller, recovery_ref: ArtifactRef, data: RecoveryInput
) -> None:
    operation = data.operation
    if not isinstance(operation, ReopenScreening):
        raise ValueError("reopening requires an explicit screening recovery request")
    layout, pool, plan = controller.layout, controller.pool, controller.plan
    request = read_document(layout, operation.request, WorkRequest)
    if request.adapter != "source_screening":
        raise ValueError("only source screening can be reopened")
    pool.register(request)
    registry = pool.ledger.registry
    registry.initialize(
        "screening-reopen.v1",
        """CREATE TABLE IF NOT EXISTS screening_reopens (
          request_digest TEXT PRIMARY KEY, request TEXT NOT NULL,
          work_id TEXT NOT NULL, previous_attempt TEXT NOT NULL,
          previous_selection TEXT NOT NULL, previous_result TEXT NOT NULL
        );""",
    )
    with (
        artifact_lock(layout, "worker-admission", blocking=False),
        artifact_lock(layout, "worker-" + request.work_id, blocking=False),
        artifact_lock(layout, "worker-services-" + request.work_id, blocking=False),
        artifact_lock(layout, "source-" + request.subject_id, blocking=False),
        artifact_lock(layout, "resource-admission", blocking=False),
    ):
        with registry.transaction() as db:
            done = db.execute(
                "SELECT request FROM screening_reopens WHERE request_digest=?",
                (data.sha256,),
            ).fetchone()
        if done is not None:
            if done["request"] != recovery_ref.canonical_bytes().decode():
                raise ValueError("screening recovery differs from its durable record")
            return
        row = pool.ledger.task(request.work_id)
        if (
            row["current_attempt"] != operation.after_attempt
            or row["state"] != "complete"
        ):
            raise ValueError("screening recovery requires the exact completed attempt")
        launch_ref, launch = pool._launch(row)
        directory = Path(launch.directory)
        pool._require_no_members(directory)
        final_ref = ArtifactRef.model_validate_json(row["result"])
        final = read_document(layout, final_ref, AttemptResult)
        pool._verify_final(final)
        if final.launch != launch_ref or final.outcome.receipt != operation.selection:
            raise ValueError(
                "screening recovery selection differs from the recorded outcome"
            )
        official = annotation_reference(layout, directory / "request.json")
        if not unfinished_services(layout, directory, official):
            raise ValueError(
                "only screening with unresolved image services can be reopened"
            )
        source = controller.collection.ledger.source(plan, request.subject_id)
        acquired = read_document(
            layout,
            ArtifactRef.model_validate_json(source["acquisition"]),
            AcquisitionReceipt,
        )
        verify_reference(layout, acquired.source.media)
        with registry.transaction() as db:
            source = db.execute(
                "SELECT state,selection,preparation FROM sources WHERE id=?",
                (request.subject_id,),
            ).fetchone()
            clips = db.execute(
                "SELECT count(*) FROM clips WHERE source_id=?", (request.subject_id,)
            ).fetchone()[0]
            active = db.execute(
                "SELECT count(*) FROM resource_requests WHERE campaign=? AND state IN ('pending','running','draining')",
                (plan.campaign_id,),
            ).fetchone()[0]
            if active:
                raise ValueError(
                    "stop active jobs and cancel pending stages before reopening screening"
                )
            if (
                source["state"] != "screened"
                or source["preparation"] is not None
                or clips
            ):
                raise ValueError(
                    "prepared or excluded sources cannot be reopened by this recovery"
                )
            if source["selection"] != operation.selection.canonical_bytes().decode():
                raise ValueError("source selection changed before recovery")
            db.execute(
                "INSERT INTO screening_reopens VALUES (?,?,?,?,?,?)",
                (
                    data.sha256,
                    recovery_ref.canonical_bytes().decode(),
                    request.work_id,
                    operation.after_attempt,
                    source["selection"],
                    row["result"],
                ),
            )
            db.execute(
                "UPDATE sources SET state='acquired',selection=NULL,reason=? WHERE id=?",
                (data.reason, request.subject_id),
            )
            db.execute(
                "UPDATE worker_tasks SET state='validation_failed',result=NULL,reason=? WHERE id=?",
                (data.reason, request.work_id),
            )
            db.execute(
                "UPDATE worker_attempts SET state='validation_failed',reason=? WHERE id=?",
                (data.reason, operation.after_attempt),
            )
