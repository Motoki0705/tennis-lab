"""Ball labels explicitly enumerate images, actors, edits, and worker telemetry."""

from __future__ import annotations

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ReviewEvidence,
)
from src.tasks.base.annotation.editing.contracts import (
    EditCommand,
    ReviewContext,
    Revision,
)
from src.tasks.base.annotation.images import read_rendered
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import references
from src.tasks.base.annotation.storage.references import distinct
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.telemetry.storage import retained_report
from src.tasks.base.annotation.workers.contracts import Launch

from .contracts import BallEdits, BallLabels, BallReviewProof
from .evidence import lineage


def retained_revision(
    layout: AnnotationLayout,
    context: ReviewContext,
    revision_ref: ArtifactRef,
) -> tuple[ArtifactRef, ...]:
    retained: list[ArtifactRef] = []
    images: dict[str, RenderedImage] = {}
    actors: set[ArtifactRef] = set()
    for reference in lineage(layout, context, revision_ref):
        revision = read_document(layout, reference, Revision)
        retained.extend((reference, *references(revision)))
        labels = read_document(layout, revision.payload, BallLabels)
        retained.extend(references(labels))
        images.update((image.sha256, image) for image in labels.images)
        actors.update(
            frame.review_actor
            for frame in labels.frames
            if frame.review_actor is not None
        )
        if revision.command is not None:
            command = read_document(layout, revision.command, EditCommand)
            retained.extend(references(command))
            edits = read_document(layout, command.changes, BallEdits)
            retained.extend(references(edits))
            images.update((image.sha256, image) for image in edits.images)
            actors.add(command.actor)
    for image in images.values():
        request, rendered = read_rendered(layout, image)
        retained.extend((*references(rendered), *references(request)))
    for actor in actors:
        retained.extend(references(read_document(layout, actor, Launch)))
    result: tuple[ArtifactRef, ...] = distinct(layout, retained)
    return result


def retained_references(
    layout: AnnotationLayout,
    context: ReviewContext,
    revision_ref: ArtifactRef,
    receipt: AdoptionReceipt,
) -> tuple[ArtifactRef, ...]:
    retained = list(retained_revision(layout, context, revision_ref))
    evidence = read_document(layout, receipt.review_evidence, ReviewEvidence)
    for reference in evidence.evidence:
        proof = read_document(layout, reference, BallReviewProof)
        retained.extend(references(proof))
        for telemetry in proof.telemetry:
            retained.extend(retained_report(layout, telemetry))
    result: tuple[ArtifactRef, ...] = distinct(layout, retained)
    return result
