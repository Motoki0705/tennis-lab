"""Input pins follow persistent requests and kernel-owned resource attempts."""

from __future__ import annotations

import sqlite3

from ..contracts import ArtifactRef, Document
from ..paths import AnnotationLayout
from ..plan import CampaignPlan
from ..storage import StorageCatalog
from ..storage.contracts import Protection
from ..storage.references import distinct, protection
from .contracts import BudgetScope


class RequestInputs(Document):
    references: tuple[ArtifactRef, ...]


class ResourceReferences:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.catalog = StorageCatalog(layout)
        self.catalog.registry.initialize(
            "resource-references.v1",
            """
            CREATE TABLE IF NOT EXISTS resource_input_protections (
                request TEXT PRIMARY KEY,
                inputs TEXT NOT NULL,
                attempt_protection TEXT
            );
            """,
        )

    def register(
        self,
        key: str,
        plan_ref: ArtifactRef,
        plan: CampaignPlan,
        scope_ref: ArtifactRef,
        scope: BudgetScope,
        inputs: tuple[ArtifactRef, ...] | None,
        *,
        database: sqlite3.Connection,
    ) -> None:
        row = database.execute(
            "SELECT inputs FROM resource_input_protections WHERE request=?", (key,)
        ).fetchone()
        if row is not None and inputs is None:
            return  # lease/attach reuses the explicitly registered input set
        document = RequestInputs(
            references=distinct(self.layout, (plan_ref, scope_ref, *(inputs or ())))
        )
        encoded = document.canonical_bytes().decode()
        if row is not None and row[0] != encoded:
            raise ValueError("resource request input references changed")
        self.catalog.protect_once(
            protection(
                self.layout,
                identity="resource-plan-v1-" + plan.sha256,
                kind="active",
                reason="resource request plan and shared budget baseline",
                roots=(plan_ref, scope_ref),
                documents=(plan, scope),
            ),
            database=database,
        )
        self.catalog.protect_once(
            protection(
                self.layout,
                identity="resource-inputs-v1-" + key,
                kind="active",
                reason="pending or resumable resource inputs awaiting downstream handoff",
                roots=document.references,
            ),
            database=database,
        )
        database.execute(
            "INSERT OR IGNORE INTO resource_input_protections VALUES (?,?,NULL)",
            (key, encoded),
        )

    def admit(self, key: str, *, database: sqlite3.Connection) -> None:
        row = database.execute(
            "SELECT inputs,attempt_protection FROM resource_input_protections WHERE request=?",
            (key,),
        ).fetchone()
        if row is None:
            raise ValueError("resource admission requires registered input references")
        if row[1] is not None:
            raise ValueError("resource attempt protection already exists")
        inputs = RequestInputs.model_validate_json(row[0])
        record = protection(
            self.layout,
            identity="resource-attempt-v1-" + key,
            kind="active",
            reason="resource lease and its remaining owned processes use these inputs",
            roots=inputs.references,
        )
        self.catalog.protect(record, database=database)
        database.execute(
            "UPDATE resource_input_protections SET attempt_protection=? WHERE request=?",
            (record.canonical_bytes().decode(), key),
        )

    def settle(self, key: str, *, database: sqlite3.Connection) -> None:
        row = database.execute(
            "SELECT attempt_protection FROM resource_input_protections WHERE request=?",
            (key,),
        ).fetchone()
        if row is None or row[0] is None:
            return  # legacy attempts were not pinned by this collector
        record = Protection.model_validate_json(row[0])
        self.catalog.release(
            record.protection_id,
            expected_digest=record.sha256,
            reason="resource lease released and all owned processes exited",
            database=database,
        )
