"""Verify a terminal empty selection without requiring a retired source video."""

from __future__ import annotations

import json

from ..artifacts import read_document, verify_reference
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, references
from ..screening.contracts import (
    PresentedImages,
    PreviewReceipt,
    ScreeningEvidence,
    ScreeningRequest,
    SourceDecision,
)
from ..storage.references import distinct
from ..workers.contracts import AttemptResult, WorkRequest
from .contracts import AcquisitionReceipt, SelectionReceipt
from .ledger import CollectionLedger


def verify_excluded(
    layout: AnnotationLayout, plan: CampaignPlan, source_id: str
) -> tuple[ArtifactRef, ArtifactRef, AcquisitionReceipt, tuple[ArtifactRef, ...]]:
    ledger = CollectionLedger(layout)
    row = ledger.source(plan, source_id)
    if row["state"] != "excluded" or row["selection"] is None or row["preparation"] is not None:
        raise ValueError("source retirement requires a committed empty selection")
    acquisition = ArtifactRef.model_validate_json(row["acquisition"])
    selection_ref = ArtifactRef.model_validate_json(row["selection"])
    acquired = read_document(layout, acquisition, AcquisitionReceipt)
    selection = read_document(layout, selection_ref, SelectionReceipt)
    if (
        selection.windows
        or not selection.shortage_reason
        or (selection.source_id, selection.source_digest, selection.policy, selection.requested_maximum)
        != (source_id, acquired.source.media.digest, plan.selection, plan.counts.max_clips_per_video)
        or acquired.configuration != plan.preparation
    ):
        raise ValueError("excluded source differs from its fixed selection inputs")
    kept = [acquisition, selection_ref, selection.evidence]
    raw = json.loads(verify_reference(layout, selection.evidence).read_bytes())
    if "request" in raw:
        evidence = read_document(layout, selection.evidence, ScreeningEvidence)
        request = read_document(layout, evidence.request, ScreeningRequest)
        decision = read_document(layout, evidence.decision, SourceDecision)
        presentation = read_document(layout, evidence.presentation, PresentedImages)
        if (
            request.plan_digest != plan.sha256
            or request.source != acquired.source
            or request.policy != plan.selection
            or decision.source_sha256 != acquired.source.media.digest
            or decision.clips
            or decision.notes != selection.shortage_reason
            or presentation.request_digest != evidence.request.digest
        ):
            raise ValueError("exclusion evidence differs from the completed selection")
        kept.extend((*references(evidence), *references(request), *references(presentation)))
        for ref in evidence.previews:
            preview = read_document(layout, ref, PreviewReceipt)
            if preview.source != acquired.source.media:
                raise ValueError("exclusion preview belongs to another source")
            kept.extend(references(preview))
        with ledger.registry.transaction() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='worker_tasks'").fetchone() is None:
                raise ValueError("exclusion requires a completed screening worker")
            matching = []
            for task in db.execute("SELECT * FROM worker_tasks WHERE campaign=?", (plan.campaign_id,)):
                work = WorkRequest.model_validate_json(task["request"])
                if work.adapter == "source_screening" and work.subject_id == source_id:
                    matching.append(task)
            if len(matching) != 1 or matching[0]["state"] != "not_applicable" or matching[0]["plan_digest"] != plan.sha256:
                raise ValueError("exclusion requires a completed screening worker")
            result_ref = ArtifactRef.model_validate_json(matching[0]["result"])
            result = read_document(layout, result_ref, AttemptResult)
            if result.outcome.status != "not_applicable" or result.outcome.receipt != selection_ref:
                raise ValueError("screening result differs from the empty selection")
            kept.extend((result_ref, *references(result)))
    else:
        # A structurally invalid video cannot provide review images. Preserve its
        # explicit probe/duration failure, rather than inventing a visual review.
        from ..screening.service import PreflightExclusion

        preflight = read_document(layout, selection.evidence, PreflightExclusion)
        if (preflight.plan_digest, preflight.acquisition, preflight.policy, preflight.reason) != (
            plan.sha256, acquisition, plan.selection, selection.shortage_reason
        ):
            raise ValueError("preflight exclusion differs from the registered selection")
    with ledger.registry.transaction() as db:
        if db.execute("SELECT 1 FROM clips WHERE source_id=? AND campaign_id=?", (source_id, plan.campaign_id)).fetchone():
            raise ValueError("excluded source unexpectedly has registered clips")
    retained = distinct(layout, tuple(ref for ref in kept if ref != acquired.source.media))
    for ref in retained:
        verify_reference(layout, ref)
    return acquisition, selection_ref, acquired, retained
