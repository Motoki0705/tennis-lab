"""Atomic partial revisions and completion, independent of prediction evidence."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext, Revision
from src.tasks.base.annotation.editing.store import RevisionConflict
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan, require_execution_layout
from src.tasks.base.annotation.registry import Registry
from src.tasks.base.annotation.storage import StorageCatalog
from src.tasks.base.annotation.storage.references import protection

from ..contracts import VideoBallAnnotation, VideoBallFrame
from ..io import read_annotation, write_annotation
from ..usage import summarize_annotation
from .contracts import AnnotationRun, EditQuery, VideoCompletion


def progress_summary(annotation: VideoBallAnnotation) -> dict[str, int | bool]:
    summary = summarize_annotation(annotation)
    return {**asdict(summary), "complete": summary.complete}


class VideoRevisionStore:
    def __init__(self, layout: AnnotationLayout, run_ref: ArtifactRef) -> None:
        self.layout, self.run_ref = layout, run_ref
        self.run = read_document(layout, run_ref, AnnotationRun)
        plan = read_document(layout, self.run.plan, LocalAnnotationPlan)
        require_execution_layout(layout, plan)
        if plan.annotation_root != str(layout.root):
            raise ValueError("blank revision plan belongs to another annotation root")
        self.context = read_document(layout, self.run.context, BlankAnnotationContext)
        # Import lazily: registration owns validation against the fixed plan inputs.
        from src.tasks.annotation.reviews import read_local_context

        self.expected = read_local_context(layout, self.run.context, self.run.plan)
        self.registry = Registry(layout)
        self.registry.initialize(
            "ball-video-revisions.v1",
            """CREATE TABLE IF NOT EXISTS ball_video_heads (
                context TEXT PRIMARY KEY, run TEXT NOT NULL, revision TEXT NOT NULL, completion TEXT);
            CREATE TABLE IF NOT EXISTS ball_video_edits (
                id TEXT PRIMARY KEY, revision TEXT NOT NULL);""",
        )
        self.catalog = StorageCatalog(layout)
        self.directory = layout.artifact_path(self.run.context).parent / "video"
        self.key = self.run.context.digest

    def initialize(self) -> ArtifactRef:
        with artifact_lock(self.layout, "blank-revision-" + self.key):
            with self.registry.transaction() as db:
                row = db.execute(
                    "SELECT run FROM ball_video_heads WHERE context=?", (self.key,)
                ).fetchone()
            if row is not None:
                return self.current()[0]
            payload = VideoBallAnnotation(
                schema_version="video_ball_annotation.v3",
                coordinate_system="native_pixels_top_left_endpoint",
                source=self.expected.source,
                target=self.expected.target,
                frame_range=self.expected.frame_range,
                frames=tuple(
                    VideoBallFrame(
                        **f.model_dump(),
                        state="unprocessed",
                        occlusion="unknown",
                        position=None,
                    )
                    for f in self.expected.frames
                ),
            )
            annotation = write_annotation(
                self.layout,
                self.directory / "initial.json",
                payload,
                expected_input=self.expected,
            )
            revision = write_document(
                self.layout,
                self.directory / "initial-revision.json",
                Revision(
                    context=self.run.context,
                    sequence=0,
                    previous=None,
                    payload=annotation,
                    command=None,
                ),
            )
            with self.registry.transaction() as db:
                self._protect(db, revision, (annotation,))
                db.execute(
                    "INSERT INTO ball_video_heads VALUES (?,?,?,NULL)",
                    (
                        self.key,
                        self.run_ref.model_dump_json(),
                        revision.model_dump_json(),
                    ),
                )
            return revision

    def _protect(
        self,
        db: sqlite3.Connection,
        root: ArtifactRef,
        dependencies: tuple[ArtifactRef, ...],
    ) -> None:
        self.catalog.protect_once(
            protection(
                self.layout,
                identity="ball-video-" + root.digest,
                kind="active",
                reason="durable annotation progress and completion",
                roots=(self.run_ref, self.run.context, root),
                documents=(self.run, self.context),
                dependencies=dependencies,
            ),
            database=db,
        )

    def current(self) -> tuple[ArtifactRef, ArtifactRef | None]:
        with self.registry.transaction() as db:
            row = db.execute(
                "SELECT * FROM ball_video_heads WHERE context=?", (self.key,)
            ).fetchone()
        if row is None or ArtifactRef.model_validate_json(row["run"]) != self.run_ref:
            raise ValueError(
                "blank revision belongs to a different immutable run/policy"
            )
        head = ArtifactRef.model_validate_json(row["revision"])
        self.annotation(head)
        completion = (
            ArtifactRef.model_validate_json(row["completion"])
            if row["completion"]
            else None
        )
        if completion is not None:
            self.validate_completion(completion, head)
        return head, completion

    def annotation(self, reference: ArtifactRef) -> VideoBallAnnotation:
        revision = read_document(self.layout, reference, Revision)
        if revision.context != self.run.context:
            raise ValueError("revision belongs to a different camera/context")
        return read_annotation(
            verify_reference(self.layout, revision.payload),
            expected_input=self.expected,
        )

    def apply(self, edit: EditQuery, actor: ArtifactRef) -> ArtifactRef:
        edit = EditQuery.model_validate_json(edit.canonical_bytes())
        if not edit.reason.strip():
            raise ValueError("edit requires a nonblank reason")
        verify_reference(self.layout, actor)
        if edit.camera_id != self.expected.source.camera_id:
            raise ValueError("edit camera differs from the assigned camera")
        indices = [f.frame_index for f in edit.rows]
        if len(indices) != len(set(indices)) or any(
            not self.expected.frame_range.start <= i < self.expected.frame_range.stop
            for i in indices
        ):
            raise ValueError("edit rows must be unique assigned source frames")
        # Do not reinterpret out-of-scope frames or another camera's supports.
        for frame in edit.rows:
            if frame.position and any(
                not self.expected.frame_range.start
                <= i
                < self.expected.frame_range.stop
                for i in frame.position.provenance.support_frames
            ):
                raise ValueError(
                    "support_frames must belong to the supplied camera input"
                )
        from src.tasks.base.annotation.contracts import content_digest

        command_id = content_digest(
            [self.run_ref.model_dump(), actor.model_dump(), edit.model_dump()]
        )
        directory = self.directory / "edits" / command_id
        with artifact_lock(self.layout, "blank-revision-" + self.key):
            with self.registry.transaction() as db:
                old = db.execute(
                    "SELECT revision FROM ball_video_edits WHERE id=?", (command_id,)
                ).fetchone()
            if old is not None:
                result = ArtifactRef.model_validate_json(old[0])
                self.annotation(result)
                return result
            head, completion = self.current()
            if completion is not None or head != edit.expected_revision:
                raise RevisionConflict(
                    "blank annotation is sealed or expected_revision is stale"
                )
            current = self.annotation(head)
            updates = {f.frame_index: f for f in edit.rows}
            payload = current.model_copy(
                update={
                    "frames": tuple(
                        updates.get(f.frame_index, f) for f in current.frames
                    )
                }
            )
            # Full validation includes source PTS, semantics, native bounds and provenance.
            annotation = write_annotation(
                self.layout,
                directory / "annotation.json",
                payload,
                expected_input=self.expected,
            )
            command = write_document(self.layout, directory / "edit.json", edit)
            previous = read_document(self.layout, head, Revision)
            result = write_document(
                self.layout,
                directory / "revision.json",
                Revision(
                    context=self.run.context,
                    sequence=previous.sequence + 1,
                    previous=head,
                    payload=annotation,
                    command=command,
                ),
            )
            with self.registry.transaction() as db:
                self._protect(db, result, (annotation, command, head, actor))
                changed = db.execute(
                    "UPDATE ball_video_heads SET revision=? WHERE context=? AND revision=? AND completion IS NULL",
                    (result.model_dump_json(), self.key, head.model_dump_json()),
                )
                if changed.rowcount != 1:
                    raise RevisionConflict("blank revision changed during save")
                db.execute(
                    "INSERT INTO ball_video_edits VALUES (?,?)",
                    (command_id, result.model_dump_json()),
                )
            return result

    def complete(
        self, revision: ArtifactRef, actor: ArtifactRef, reason: str
    ) -> ArtifactRef:
        if not reason.strip():
            raise ValueError("completion requires a nonblank reason")
        with artifact_lock(self.layout, "blank-revision-" + self.key):
            head, old = self.current()
            if head != revision:
                raise RevisionConflict("completion must name the current revision")
            if old is not None:
                return old
            if not summarize_annotation(self.annotation(head)).complete:
                raise ValueError("completion contains unprocessed frames")
            payload = read_document(self.layout, head, Revision).payload
            path = self.directory / "completion.json"
            if path.exists():
                result = annotation_reference(self.layout, path)
                receipt = self.validate_completion(result, head)
            else:
                receipt = VideoCompletion(
                    run=self.run_ref,
                    context=self.run.context,
                    revision=head,
                    annotation=payload,
                    actor=actor,
                    reason=reason,
                )
                result = write_document(self.layout, path, receipt)
            with self.registry.transaction() as db:
                self._protect(db, result, (head, payload, receipt.actor))
                db.execute(
                    "UPDATE ball_video_heads SET completion=? WHERE context=? AND revision=?",
                    (result.model_dump_json(), self.key, head.model_dump_json()),
                )
            return result

    def validate_completion(
        self, reference: ArtifactRef, head: ArtifactRef
    ) -> VideoCompletion:
        receipt: VideoCompletion = read_document(
            self.layout, reference, VideoCompletion
        )
        revision = read_document(self.layout, head, Revision)
        if (receipt.run, receipt.context, receipt.revision, receipt.annotation) != (
            self.run_ref,
            self.run.context,
            head,
            revision.payload,
        ):
            raise ValueError("completion differs from the pinned camera/run/revision")
        verify_reference(self.layout, receipt.actor)
        read_annotation(
            verify_reference(self.layout, receipt.annotation),
            expected_input=self.expected,
            require_complete=True,
        )
        return receipt
