"""Cross-campaign immutable work identity and mutable attempt state."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..artifacts import annotation_reference, read_document, verify_reference
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references, require_execution_layout, save_plan
from ..registry import Registry
from ..telemetry.capture import verify_report
from .contracts import AttemptEvidence, AttemptResult, ExitReceipt, Launch, WorkRequest
from .storage import WorkerReferences

SCHEMA = """
CREATE TABLE IF NOT EXISTS worker_tasks (
 id TEXT PRIMARY KEY, campaign TEXT NOT NULL, plan_digest TEXT NOT NULL,
 request TEXT NOT NULL, state TEXT NOT NULL, current_attempt TEXT,
 result TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS worker_attempts (
 id TEXT PRIMARY KEY, work_id TEXT NOT NULL REFERENCES worker_tasks(id),
 number INTEGER NOT NULL, state TEXT NOT NULL, launch TEXT NOT NULL,
 reason TEXT, UNIQUE (work_id,number)
);
CREATE TABLE IF NOT EXISTS worker_campaign_stops (
 campaign TEXT PRIMARY KEY, reason TEXT NOT NULL
);
"""


class WorkerBusy(RuntimeError):
    """The exact work or a shared worker capacity is still owned."""


class WorkerStopped(RuntimeError):
    """No automatic retry of failed, deferred, cancelled or quota-limited work."""


class WorkerLedger:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.registry = Registry(layout)
        self.registry.initialize("workers.v1", SCHEMA)
        self.storage = WorkerReferences(layout)

    def register(
        self,
        plan: CampaignPlan,
        request: WorkRequest,
        *,
        dependencies: tuple[ArtifactRef, ...],
    ) -> None:
        require_execution_layout(self.layout, plan)
        plan_ref = annotation_reference(self.layout, save_plan(self.layout, plan))
        for reference in references(request):
            verify_reference(self.layout, reference)
        stage = next(
            (stage for stage in plan.resources.stages if stage.stage == request.stage),
            None,
        )
        if stage is None or stage.kind != "codex":
            raise ValueError("worker must use an explicitly allocated Codex stage")
        with self.registry.transaction() as database:
            row = database.execute(
                "SELECT * FROM worker_tasks WHERE id=?", (request.work_id,)
            ).fetchone()
            if row is not None:
                self.require(row, plan, request)
            else:
                database.execute(
                    "INSERT INTO worker_tasks VALUES (?,?,?,?, 'ready',NULL,NULL,NULL)",
                    (
                        request.work_id,
                        plan.campaign_id,
                        plan.sha256,
                        request.canonical_bytes().decode(),
                    ),
                )
            self.storage.register(
                plan_ref, plan, request, dependencies, database=database
            )

    @staticmethod
    def require(row: sqlite3.Row, plan: CampaignPlan, request: WorkRequest) -> None:
        if (row["campaign"], row["plan_digest"], row["request"]) != (
            plan.campaign_id,
            plan.sha256,
            request.canonical_bytes().decode(),
        ):
            raise ValueError(
                "same task/subject is already owned by another immutable campaign/request"
            )

    def task(self, work_id: str) -> dict[str, Any]:
        with self.registry.transaction() as database:
            row = database.execute(
                "SELECT * FROM worker_tasks WHERE id=?", (work_id,)
            ).fetchone()
            if row is None:
                raise KeyError(work_id)
            return dict(row)

    def resume_campaign(self, campaign: str) -> None:
        with self.registry.transaction() as database:
            database.execute(
                "DELETE FROM worker_campaign_stops WHERE campaign=?", (campaign,)
            )

    def history(
        self, plan: CampaignPlan, request: WorkRequest
    ) -> tuple[AttemptEvidence, ...]:
        """Read immutable attempt files, never the replaceable task.result column."""
        with self.registry.transaction() as database:
            task = database.execute(
                "SELECT * FROM worker_tasks WHERE id=?", (request.work_id,)
            ).fetchone()
            if task is None:
                raise KeyError(request.work_id)
            self.require(task, plan, request)
            rows = database.execute(
                "SELECT * FROM worker_attempts WHERE work_id=? ORDER BY number",
                (request.work_id,),
            ).fetchall()
        result = []
        for row in rows:
            launch_ref = ArtifactRef.model_validate_json(row["launch"])
            launch = read_document(self.layout, launch_ref, Launch)
            directory = self.layout.path(
                "campaigns",
                plan.campaign_id,
                "workers",
                request.work_id,
                f"attempt-{row['number']:04d}",
            )
            if (
                launch.request != request
                or launch.work_id != request.work_id
                or launch.attempt != row["id"]
                or read_document(self.layout, launch.plan, CampaignPlan) != plan
                or Path(launch.directory) != directory
                or self.layout.artifact_path(launch_ref) != directory / "launch.json"
            ):
                raise ValueError("attempt history differs from its registered launch")

            def optional(name: str, directory: Path = directory) -> ArtifactRef | None:
                path = directory / name
                return (
                    annotation_reference(self.layout, path) if path.exists() else None
                )

            execution_ref = optional("exit.json")
            if execution_ref is not None:
                execution = read_document(self.layout, execution_ref, ExitReceipt)
                if execution.launch_digest != launch_ref.digest:
                    raise ValueError("attempt exit belongs to another launch")
                for ref in references(execution):
                    verify_reference(self.layout, ref)
            telemetry_pointer = optional("telemetry.ref.json")
            telemetry = (
                read_document(self.layout, telemetry_pointer, ArtifactRef)
                if telemetry_pointer
                else None
            )
            final_ref = optional("result.receipt.json")
            if final_ref is not None:
                final = read_document(self.layout, final_ref, AttemptResult)
                if (final.launch, final.exit) != (launch_ref, execution_ref):
                    raise ValueError("attempt result belongs to another launch/exit")
                for ref in references(final):
                    verify_reference(self.layout, ref)
                if telemetry is not None and final.telemetry != telemetry:
                    raise ValueError("attempt has conflicting telemetry references")
                telemetry = final.telemetry
            if telemetry is not None:
                report = verify_report(self.layout, telemetry)
                if (report.launch, report.execution) != (launch_ref, execution_ref):
                    raise ValueError("attempt telemetry belongs to another launch/exit")
            result.append(
                AttemptEvidence(
                    number=row["number"],
                    state=row["state"],
                    reason=row["reason"],
                    launch=launch_ref,
                    execution=execution_ref,
                    result=final_ref,
                    telemetry=telemetry,
                )
            )
        return tuple(result)

    def stop_campaign(self, campaign: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("campaign stop requires a reason")
        with self.registry.transaction() as database:
            database.execute(
                "INSERT OR REPLACE INTO worker_campaign_stops VALUES (?,?)",
                (campaign, reason),
            )
