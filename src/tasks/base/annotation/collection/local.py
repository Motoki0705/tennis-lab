"""Retained local media and opaque direct inputs in the shared annotation registry."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
)
from ..contracts import ArtifactRef, ClipManifest, Document, MediaFingerprint
from ..editing.contracts import BlankAnnotationContext
from ..paths import AnnotationLayout
from ..plan import LocalAnnotationPlan, file_digest, references
from ..workers.contracts import WorkRequest
from ..workers.ledger import WorkerLedger
from .ledger import CollectionLedger


class LocalInputRegistration(Document):
    context: ArtifactRef
    request: ArtifactRef


def retain_local_source(
    layout: AnnotationLayout, path: Path, expected: MediaFingerprint
) -> ArtifactRef:
    """Copy original bytes into owned storage; never unlink the caller's file."""
    destination = layout.path("sources", "blobs", expected.digest)
    with artifact_lock(layout, "source-content-" + expected.digest):
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                staged = Path(temporary) / "source"
                shutil.copyfile(path, staged)
                if (
                    staged.stat().st_size != expected.size_bytes
                    or file_digest(staged) != expected.digest
                ):
                    raise ValueError(
                        "local source changed while retaining its pinned bytes"
                    )
                with staged.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.rename(staged, destination)
                sync_directory(destination.parent)
        reference = annotation_reference(layout, destination)
        if (reference.digest, reference.size_bytes) != (
            expected.digest,
            expected.size_bytes,
        ):
            raise ValueError("retained original source bytes changed")
        return reference


class LocalInputLedger:
    """Local inputs use campaign and worker ledgers without prediction task rows."""

    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.collection = CollectionLedger(layout)
        self.registry = self.collection.registry
        self.registry.initialize(
            "local-annotation-inputs.v1",
            """
            CREATE TABLE IF NOT EXISTS local_annotation_inputs (
                work_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                registration TEXT NOT NULL
            );
            """,
        )
        self.workers = WorkerLedger(layout)

    def register(
        self, plan: LocalAnnotationPlan, entry: LocalInputRegistration
    ) -> None:
        context = read_document(self.layout, entry.context, BlankAnnotationContext)
        request = read_document(self.layout, entry.request, WorkRequest)
        saved_plan = read_document(self.layout, context.plan, LocalAnnotationPlan)
        if (
            saved_plan != plan
            or context.task != plan.task
            or request.payload != entry.context
        ):
            raise ValueError(
                "local input context/request differs from its immutable plan"
            )
        manifest = read_document(self.layout, context.manifest, ClipManifest)
        if request.subject_id != manifest.clip_id:
            raise ValueError("local work subject differs from its image manifest")
        if set(request.inputs) != {
            context.manifest,
            context.task_input,
            context.source_media,
        }:
            raise ValueError("local work inputs differ from its blank context")
        dependencies = (*references(context), manifest.media)
        for reference in dependencies:
            verify_reference(self.layout, reference)
        self.collection.register_campaign(plan)
        with self.registry.transaction() as database:
            row = database.execute(
                "SELECT campaign_id,registration FROM local_annotation_inputs WHERE work_id=?",
                (request.work_id,),
            ).fetchone()
            encoded = entry.canonical_bytes().decode()
            if row is not None and tuple(row) != (plan.campaign_id, encoded):
                raise ValueError(
                    "local input already belongs to a different plan/input version"
                )
            database.execute(
                "INSERT OR IGNORE INTO local_annotation_inputs VALUES (?,?,?)",
                (request.work_id, plan.campaign_id, encoded),
            )
        # Registration and GC pins reconcile on replay, including a stop between commits.
        self.workers.register(plan, request, dependencies=dependencies)

    def entries(self, plan: LocalAnnotationPlan) -> tuple[LocalInputRegistration, ...]:
        with self.registry.transaction() as database:
            self.collection._require_plan(database, plan)
            rows = database.execute(
                "SELECT registration FROM local_annotation_inputs WHERE campaign_id=? ORDER BY work_id",
                (plan.campaign_id,),
            ).fetchall()
        return tuple(LocalInputRegistration.model_validate_json(row[0]) for row in rows)
