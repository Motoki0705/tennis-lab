"""Per-clip immutable revisions, batch validation, CAS, and replayable edit receipts."""

from __future__ import annotations

from pathlib import Path

from ..artifacts import (
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from ..contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
    ReviewEvidence,
    content_digest,
)
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, load_plan, references
from ..predictions import PredictionReceipt
from ..registry import Registry
from ..storage.catalog import StorageCatalog
from .contracts import EditCommand, EditReceipt, ReviewAdapter, ReviewContext, Revision
from .storage import retained_adoption

_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_revision_heads (
    id TEXT PRIMARY KEY, context TEXT NOT NULL, revision TEXT NOT NULL, adoption TEXT
);
CREATE TABLE IF NOT EXISTS annotation_edits (
    id TEXT PRIMARY KEY, head_id TEXT NOT NULL, receipt TEXT NOT NULL,
    FOREIGN KEY(head_id) REFERENCES annotation_revision_heads(id)
);
"""


class RevisionConflict(ValueError):
    pass


class RevisionStore:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.registry = Registry(layout)
        self.registry.initialize("annotation-revisions-v1", _SCHEMA)
        self.catalog = StorageCatalog(layout)

    def context(
        self, value: ReviewContext, adapter: ReviewAdapter
    ) -> tuple[CampaignPlan, ClipManifest, str]:
        plan = load_plan(verify_reference(self.layout, value.plan))
        task = next((t for t in plan.tasks if t.task == value.task), None)
        if (
            plan.annotation_root != str(self.layout.root)
            or not plan.allows_adoption
            or task is None
            or task.review is None
            or adapter.task != value.task
        ):
            raise ValueError(
                "editing requires a production task with an evidence-backed review policy"
            )
        for ref in references(task.review):
            verify_reference(self.layout, ref)
        manifest = read_document(self.layout, value.manifest, ClipManifest)
        prediction = read_document(self.layout, value.prediction, PredictionReceipt)
        ctx = prediction.context
        if (ctx.task, ctx.manifest, ctx.clip_id, ctx.model, ctx.configuration) != (
            value.task,
            value.manifest,
            manifest.clip_id,
            task.model,
            task.inference,
        ) or prediction.prediction.request_id != ctx.request_id:
            raise ValueError("review prediction differs from its clip/campaign/model")
        verify_reference(self.layout, prediction.prediction.artifact)
        key = content_digest([value.task, manifest.clip_id])
        with self.registry.transaction() as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {"campaigns", "annotation_predictions", "annotation_tasks"}.issubset(tables):
                raise ValueError("review needs a registered new-clip prediction")
            row = db.execute(
                "SELECT t.campaign_id,p.receipt,c.digest AS plan_digest "
                "FROM annotation_tasks t "
                "JOIN annotation_predictions p ON t.id=p.task_id "
                "JOIN campaigns c ON c.id=t.campaign_id WHERE t.id=?",
                (key,),
            ).fetchone()
        if (
            row is None
            or row["campaign_id"] != plan.campaign_id
            or ArtifactRef.model_validate_json(row["receipt"]) != value.prediction
        ):
            raise ValueError(
                "review is not bound to this campaign's registered prediction"
            )
        if row["plan_digest"] != plan.sha256:
            raise ValueError("review plan differs from the registered campaign plan")
        return plan, manifest, key

    def _directory(self, context: ReviewContext, manifest: ClipManifest) -> Path:
        result: Path = self.layout.path("annotations", context.task, manifest.clip_id)
        return result

    def _head(
        self, key: str, context: ArtifactRef
    ) -> tuple[ArtifactRef, ArtifactRef | None]:
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT * FROM annotation_revision_heads WHERE id=?", (key,)
            ).fetchone()
        if row is None or ArtifactRef.model_validate_json(row["context"]) != context:
            raise ValueError(
                "review context is not registered, or already belongs to another campaign"
            )
        return ArtifactRef.model_validate_json(
            row["revision"]
        ), ArtifactRef.model_validate_json(row["adoption"]) if row["adoption"] else None

    def bind_context(
        self, context: ReviewContext, adapter: ReviewAdapter
    ) -> ArtifactRef:
        """Pin a canonical request context without initializing or adopting labels."""
        _, manifest, _ = self.context(context, adapter)
        result: ArtifactRef = write_document(
            self.layout, self._directory(context, manifest) / "context.json", context
        )
        return result

    def initialize(self, context: ReviewContext, adapter: ReviewAdapter) -> ArtifactRef:
        _, manifest, key = self.context(context, adapter)
        directory = self._directory(context, manifest)
        with artifact_lock(self.layout, "review-" + key, blocking=False):
            context_ref = write_document(
                self.layout, directory / "context.json", context
            )
            with self.registry.transaction() as db:
                row = db.execute(
                    "SELECT * FROM annotation_revision_heads WHERE id=?", (key,)
                ).fetchone()
                state = db.execute(
                    "SELECT state FROM annotation_tasks WHERE id=?", (key,)
                ).fetchone()
            if row is not None:
                return self._head(key, context_ref)[0]
            if state is None or state["state"] != "predicted":
                raise ValueError(
                    "initial revision requires a new, unreviewed prediction"
                )
            verify_reference(self.layout, manifest.media)
            payload = adapter.initialize(self.layout, context, manifest)
            adapter.validate(self.layout, context, manifest, payload)
            payload_ref = write_document(
                self.layout, directory / "initial-payload.json", payload
            )
            revision = Revision(
                context=context_ref,
                sequence=0,
                previous=None,
                payload=payload_ref,
                command=None,
            )
            ref = write_document(
                self.layout, directory / "initial-revision.json", revision
            )
            with self.registry.transaction() as db:
                db.execute(
                    "INSERT INTO annotation_revision_heads VALUES (?,?,?,NULL)",
                    (key, context_ref.model_dump_json(), ref.model_dump_json()),
                )
                db.execute(
                    "UPDATE annotation_tasks SET state='reviewing' WHERE id=?", (key,)
                )
            return ref

    def current(
        self, context_ref: ArtifactRef, adapter: ReviewAdapter
    ) -> tuple[ArtifactRef, ArtifactRef | None]:
        context = read_document(self.layout, context_ref, ReviewContext)
        _, _, key = self.context(context, adapter)
        result = self._head(key, context_ref)
        for ref in result:
            if ref is not None:
                verify_reference(self.layout, ref)
        return result

    def apply(self, command: EditCommand, adapter: ReviewAdapter) -> ArtifactRef:
        if not command.reason.strip():
            raise ValueError("edit needs a nonempty reason")
        context = read_document(self.layout, command.context, ReviewContext)
        _, manifest, key = self.context(context, adapter)
        for ref in references(command):
            verify_reference(self.layout, ref)
        directory = self._directory(context, manifest) / "edits" / command.sha256
        with artifact_lock(self.layout, "review-" + key, blocking=False):
            with self.registry.transaction() as db:
                old = db.execute(
                    "SELECT receipt FROM annotation_edits WHERE id=?", (command.sha256,)
                ).fetchone()
            if old is not None:
                receipt = EditReceipt.model_validate_json(old["receipt"])
                for ref in references(receipt):
                    verify_reference(self.layout, ref)
                return write_document(self.layout, directory / "receipt.json", receipt)
            head, adoption = self._head(key, command.context)
            if adoption is not None:
                raise RevisionConflict(
                    "adopted annotations are sealed; new edits are forbidden"
                )
            if head != command.expected_revision:
                raise RevisionConflict(
                    "annotation revision changed; read the current revision before editing"
                )
            previous = read_document(self.layout, head, Revision)
            verify_reference(self.layout, previous.payload)
            result = adapter.apply(
                self.layout, context, manifest, previous.payload, command
            )
            adapter.validate(self.layout, context, manifest, result.payload)
            verify_reference(self.layout, previous.payload)
            summary = result.summary
            if (
                len(summary.examples) > summary.changed_rows
                or any(
                    a.stop > b.start
                    for a, b in zip(
                        summary.changed_ranges, summary.changed_ranges[1:], strict=False
                    )
                )
                or any(r.stop > len(manifest.frames) for r in summary.changed_ranges)
                or any(
                    item.frame_index >= len(manifest.frames)
                    for item in summary.examples
                )
            ):
                raise ValueError("edit summary lies outside its frame/row bounds")
            command_ref = write_document(
                self.layout, directory / "command.json", command
            )
            payload_ref = write_document(
                self.layout, directory / "payload.json", result.payload
            )
            revision = Revision(
                context=command.context,
                sequence=previous.sequence + 1,
                previous=head,
                payload=payload_ref,
                command=command_ref,
            )
            revision_ref = write_document(
                self.layout, directory / "revision.json", revision
            )
            receipt = EditReceipt(
                command=command_ref,
                previous=head,
                revision=revision_ref,
                summary=summary,
            )
            with self.registry.transaction() as db:
                changed = db.execute(
                    "UPDATE annotation_revision_heads SET revision=? WHERE id=? AND revision=? AND adoption IS NULL",
                    (revision_ref.model_dump_json(), key, head.model_dump_json()),
                )
                if changed.rowcount != 1:
                    raise RevisionConflict(
                        "annotation revision changed during the edit"
                    )
                db.execute(
                    "INSERT INTO annotation_edits VALUES (?,?,?)",
                    (command.sha256, key, receipt.model_dump_json()),
                )
            # A lost response/file write after commit reuses the stored receipt.
            return write_document(self.layout, directory / "receipt.json", receipt)

    def seal(
        self,
        revision_ref: ArtifactRef,
        adoption_ref: ArtifactRef,
        adapter: ReviewAdapter,
    ) -> ArtifactRef:
        revision = read_document(self.layout, revision_ref, Revision)
        context = read_document(self.layout, revision.context, ReviewContext)
        plan, manifest, key = self.context(context, adapter)
        receipt = read_document(self.layout, adoption_ref, AdoptionReceipt)
        task = next(t for t in plan.tasks if t.task == context.task)
        assert task.review is not None
        if (receipt.task, receipt.clip_id, receipt.revision, receipt.policy) != (
            context.task,
            manifest.clip_id,
            revision_ref,
            task.review.policy,
        ):
            raise ValueError("adoption does not match the current clip/revision/policy")
        evidence = read_document(self.layout, receipt.review_evidence, ReviewEvidence)
        if (evidence.clip_id, evidence.revision, evidence.prediction) != (
            manifest.clip_id,
            revision_ref,
            context.prediction,
        ):
            raise ValueError(
                "review evidence belongs to another clip/revision/prediction"
            )
        for ref in (*references(receipt), *references(evidence)):
            verify_reference(self.layout, ref)
        adapter.validate_adoption(self.layout, context, manifest, revision_ref, receipt)
        retained = retained_adoption(self.layout, revision_ref, adoption_ref, adapter)
        with artifact_lock(self.layout, "review-" + key, blocking=False):
            head, old = self._head(key, revision.context)
            if head != revision_ref or (old is not None and old != adoption_ref):
                raise RevisionConflict(
                    "adoption conflicts with the current revision or receipt"
                )
            with self.registry.transaction() as db:
                self.catalog.protect(retained, database=db)
                db.execute(
                    "UPDATE annotation_revision_heads SET adoption=? WHERE id=?",
                    (adoption_ref.model_dump_json(), key),
                )
                db.execute(
                    "UPDATE annotation_tasks SET state='adopted' WHERE id=?", (key,)
                )
        return adoption_ref

    def changes_since(
        self, current: ArtifactRef, previous: ArtifactRef, *, limit: int = 20
    ) -> tuple[tuple[ArtifactRef, ...], ArtifactRef | None]:
        """Newest-first command refs and continuation cursor; never inline full rows."""
        if not 1 <= limit <= 50:
            raise ValueError("history page limit must be in 1..50")
        ancestor = read_document(self.layout, previous, Revision)
        cursor = current
        commands: list[ArtifactRef] = []
        while cursor != previous:
            revision = read_document(self.layout, cursor, Revision)
            if (
                revision.context != ancestor.context
                or revision.sequence <= ancestor.sequence
                or revision.previous is None
                or revision.command is None
            ):
                raise ValueError(
                    "history revisions are not from one increasing lineage"
                )
            commands.append(revision.command)
            cursor = revision.previous
            if len(commands) == limit:
                return tuple(commands), cursor if cursor != previous else None
        return tuple(commands), None
