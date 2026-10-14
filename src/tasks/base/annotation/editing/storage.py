"""Retain sealed revision lineage and task-enumerated evidence as one protection."""

from __future__ import annotations

from ..artifacts import read_document
from ..contracts import AdoptionReceipt, ArtifactRef, ClipManifest, ReviewEvidence
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references
from ..predictions import PredictionReceipt
from ..storage.contracts import Protection
from ..storage.references import protection
from .contracts import EditCommand, ReviewAdapter, ReviewContext, Revision


def retained_adoption(
    layout: AnnotationLayout,
    revision_ref: ArtifactRef,
    adoption_ref: ArtifactRef,
    adapter: ReviewAdapter,
) -> Protection:
    revision = read_document(layout, revision_ref, Revision)
    context = read_document(layout, revision.context, ReviewContext)
    receipt = read_document(layout, adoption_ref, AdoptionReceipt)
    documents = (
        receipt,
        context,
        read_document(layout, context.plan, CampaignPlan),
        read_document(layout, context.manifest, ClipManifest),
        read_document(layout, context.prediction, PredictionReceipt),
        read_document(layout, receipt.review_evidence, ReviewEvidence),
    )
    retained: list[ArtifactRef] = []
    seen: set[ArtifactRef] = set()
    current: ArtifactRef | None = revision_ref
    sequence = revision.sequence
    while current is not None:
        if current in seen:
            raise ValueError("adopted revision lineage contains a cycle")
        seen.add(current)
        node = read_document(layout, current, Revision)
        if node.context != revision.context or node.sequence != sequence:
            raise ValueError("adopted revision lineage changed context or sequence")
        retained.extend((current, *references(node)))
        if node.command is not None:
            command = read_document(layout, node.command, EditCommand)
            if (
                command.context != node.context
                or command.expected_revision != node.previous
            ):
                raise ValueError("adopted edit command differs from revision lineage")
            retained.extend(references(command))
        sequence, current = sequence - 1, node.previous
    retained.extend(adapter.retained_references(layout, context, revision_ref, receipt))
    return protection(
        layout,
        identity="adoption-v1-" + adoption_ref.digest,
        kind="adoption",
        reason="sealed annotation, full revision lineage, and task-enumerated review evidence",
        roots=(adoption_ref,),
        documents=documents,
        dependencies=tuple(retained),
    )
