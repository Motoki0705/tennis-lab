"""Keep registered work inputs and process-owned attempt inputs out of GC plans."""

from __future__ import annotations

import sqlite3

from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import CampaignPlan
from ..storage import StorageCatalog
from ..storage.contracts import Protection
from ..storage.references import protection
from .contracts import Launch, WorkRequest


class WorkerReferences:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.catalog = StorageCatalog(layout)
        self.catalog.registry.initialize(
            "worker-references.v1",
            """
            CREATE TABLE IF NOT EXISTS worker_attempt_protections (
                attempt TEXT PRIMARY KEY,
                protection TEXT NOT NULL
            );
            """,
        )

    def register(
        self,
        plan_ref: ArtifactRef,
        plan: CampaignPlan,
        request: WorkRequest,
        dependencies: tuple[ArtifactRef, ...],
        *,
        database: sqlite3.Connection,
    ) -> None:
        self.catalog.protect_once(
            protection(
                self.layout,
                identity="worker-plan-v1-" + plan.sha256,
                kind="active",
                reason="registered worker plan and immutable campaign inputs",
                roots=(plan_ref,),
                documents=(plan,),
            ),
            database=database,
        )
        self.catalog.protect_once(
            protection(
                self.layout,
                identity="worker-inputs-v1-" + request.sha256,
                kind="active",
                reason="registered work inputs retained for continuation and downstream handoff",
                roots=(plan_ref,),
                documents=(request,),
                dependencies=dependencies,
            ),
            database=database,
        )

    def launch(
        self,
        reference: ArtifactRef,
        launch: Launch,
        *,
        database: sqlite3.Connection,
    ) -> None:
        record = protection(
            self.layout,
            identity="worker-attempt-v1-" + reference.digest,
            kind="active",
            reason="worker process and pending finalization own these attempt inputs",
            roots=(reference,),
            documents=(launch,),
        )
        self.catalog.protect(record, database=database)
        database.execute(
            "INSERT INTO worker_attempt_protections VALUES (?,?)",
            (launch.attempt, record.canonical_bytes().decode()),
        )

    def settle(self, attempt: str, *, database: sqlite3.Connection) -> None:
        """Caller holds the work lock and has proved all owned processes exited."""
        row = database.execute(
            "SELECT protection FROM worker_attempt_protections WHERE attempt=?",
            (attempt,),
        ).fetchone()
        if row is None:
            # Older attempts have no pin; do not claim to have backfilled them.
            return
        record = Protection.model_validate_json(row[0])
        self.catalog.release(
            record.protection_id,
            expected_digest=record.sha256,
            reason="owned processes exited and worker finalization settled",
            database=database,
        )
