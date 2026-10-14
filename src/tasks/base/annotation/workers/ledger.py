"""Cross-campaign immutable work identity and mutable attempt state."""

from __future__ import annotations

import sqlite3
from typing import Any

from ..artifacts import verify_reference
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references, save_plan
from ..registry import Registry
from .contracts import WorkRequest

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

    def register(self, plan: CampaignPlan, request: WorkRequest) -> None:
        save_plan(self.layout, plan)
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

    def stop_campaign(self, campaign: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("campaign stop requires a reason")
        with self.registry.transaction() as database:
            database.execute(
                "INSERT OR REPLACE INTO worker_campaign_stops VALUES (?,?)",
                (campaign, reason),
            )
