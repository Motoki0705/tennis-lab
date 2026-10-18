"""Human-selected storage policy with immutable measurement and reader evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ...artifacts import artifact_lock, read_document, verify_reference, write_document
from ...contracts import ArtifactRef, Document, Identifier
from ...paths import AnnotationLayout
from ..catalog import StorageCatalog
from ..contracts import Inventory
from ..references import protection
from .contracts import CapturedStore, ComparisonReport, ComparisonRequest, StoreSpec


class PolicyChoice(Document):
    format: Literal["native", "image_records"]
    shards: Literal["copy", "share"]
    cache: Literal["decode", "private_rgb", "shared_rgb"]


class PolicyDecision(Document):
    """A choice for a declared use, not a change to dataset or training readers."""

    schema_version: Literal["annotation.storage.policy-decision.v1"] = (
        "annotation.storage.policy-decision.v1"
    )
    decision_id: Identifier
    report: ArtifactRef
    selected: PolicyChoice
    reason: str = Field(min_length=1)
    usage_scope: str = Field(min_length=1)
    retention_reason: str = Field(min_length=1)
    reader_compatibility: tuple[ArtifactRef, ...] = Field(min_length=1)


def record_decision(layout: AnnotationLayout, decision: PolicyDecision) -> ArtifactRef:
    """Persist a deliberate choice and protect its evidence, without auto-adoption."""
    if any(
        not text.strip()
        for text in (decision.reason, decision.usage_scope, decision.retention_reason)
    ):
        raise ValueError("storage policy requires explicit reasons and usage scope")
    catalog = StorageCatalog(layout)
    with artifact_lock(layout, "storage-policy-" + decision.decision_id):
        report = read_document(layout, decision.report, ComparisonReport)
        request = read_document(layout, report.request, ComparisonRequest)
        captured = read_document(layout, request.source, CapturedStore)
        specification = read_document(layout, captured.specification, StoreSpec)
        choices = [
            PolicyChoice(format=p.format, shards=p.shards, cache=p.cache)
            for p in report.policies
        ]
        if len({choice.sha256 for choice in choices}) != len(choices):
            raise ValueError("comparison contains duplicate policies")
        if decision.selected not in choices:
            raise ValueError("selected storage policy was not measured")
        if [row.row for row in report.rows] != list(request.rows) or any(
            len(policy.attempts) != request.attempts
            or any(len(attempt.passes) != request.passes for attempt in policy.attempts)
            for policy in report.policies
        ):
            raise ValueError("comparison does not match its fixed rows/attempts/passes")
        for reference in (
            report.source_inventory,
            report.converted_inventory,
            *(policy.inventory for policy in report.policies),
        ):
            read_document(layout, reference, Inventory)
        for reference in decision.reader_compatibility:
            verify_reference(layout, reference)
        saved = write_document(
            layout,
            layout.path("storage_decisions", decision.decision_id + ".json"),
            decision,
        )
        catalog.protect(
            protection(
                layout,
                identity="storage-policy-v1-" + decision.decision_id,
                kind="adoption",
                reason="Explicit storage policy choice and its measured/reader evidence",
                roots=(saved,),
                documents=(decision, report, request, captured, specification),
            )
        )
        return saved
