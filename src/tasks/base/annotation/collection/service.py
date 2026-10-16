"""Reconcile immutable acquisition/preparation receipts into the shared ledger."""

from __future__ import annotations

from collections.abc import Iterable

from ..artifacts import annotation_reference, artifact_lock, read_document
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import CampaignPlan
from .acquisition import acquire_source
from .contracts import (
    AcquisitionReceipt,
    PreparationConfig,
    RegistrationResult,
    SourceCandidate,
)
from .ledger import CollectionLedger
from .preparation import prepare_source


class SourceExcluded(ValueError):
    """A recorded exclusion, not a successful acquisition/preparation."""


def round_robin(candidates: Iterable[SourceCandidate]) -> list[SourceCandidate]:
    channels: dict[tuple[str, str], list[SourceCandidate]] = {}
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.source_id in seen:
            raise ValueError("selection repeats a canonical video identity")
        seen.add(candidate.source_id)
        channels.setdefault((candidate.provider, candidate.channel_id), []).append(
            candidate
        )
    return [
        items[index]
        for index in range(max(map(len, channels.values()), default=0))
        for items in channels.values()
        if index < len(items)
    ]


class Collection:
    def __init__(self, layout: AnnotationLayout, plan: CampaignPlan) -> None:
        if plan.phase == "benchmark" or layout.review_namespace is not None:
            raise ValueError(
                "benchmark review reuses frozen inputs and cannot collect media"
            )
        self.layout, self.plan = layout, plan
        self.ledger = CollectionLedger(layout)
        self.ledger.register_campaign(plan)
        self.config = read_document(layout, plan.preparation, PreparationConfig)

    def register(
        self, candidates: Iterable[SourceCandidate]
    ) -> list[RegistrationResult]:
        return [
            self.ledger.register_source(self.plan, candidate)
            for candidate in round_robin(candidates)
        ]

    def acquire(self, source_id: str, *, retry: bool = False) -> ArtifactRef:
        with artifact_lock(self.layout, "source-" + source_id, blocking=False):
            row = self.ledger.source(self.plan, source_id)
            if row["state"] == "excluded":
                raise SourceExcluded(row["reason"])
            candidate = SourceCandidate.model_validate_json(row["candidate"])
            receipt_path = self.layout.path(
                "sources", source_id, self.plan.preparation.digest, "acquisition.json"
            )
            started = self.ledger.begin_acquisition(
                self.plan,
                source_id,
                self.config.maximum_unprepared_sources,
                retry=retry,
            )
            if receipt_path.exists():
                reference = annotation_reference(self.layout, receipt_path)
            else:
                if not started:
                    raise ValueError("registered acquisition receipt is missing")
                try:
                    reference = acquire_source(
                        self.layout, candidate, self.plan.preparation, self.config
                    )
                except Exception as error:
                    self.ledger.acquisition_failed(
                        self.plan, source_id, f"{type(error).__name__}: {error}"
                    )
                    raise
            if not self.ledger.record_acquisition(self.plan, reference):
                raise SourceExcluded(self.ledger.source(self.plan, source_id)["reason"])
            return reference

    def select(self, reference: ArtifactRef) -> None:
        self.ledger.record_selection(self.plan, reference)

    def exclude_preparation(self, source_id: str, *, reason: str) -> None:
        """Record a permanent preparation failure and release its admission slots."""
        with artifact_lock(self.layout, "source-" + source_id, blocking=False):
            row = self.ledger.source(self.plan, source_id)
            receipt_path = self.layout.path(
                "sources", source_id, self.plan.preparation.digest, "prepared.json"
            )
            if row["preparation"] is not None or receipt_path.exists():
                raise ValueError("cannot exclude a source with a published preparation")
            self.ledger.exclude_preparation(self.plan, source_id, reason)

    def prepare(self, source_id: str) -> ArtifactRef:
        with artifact_lock(self.layout, "source-" + source_id, blocking=False):
            row = self.ledger.source(self.plan, source_id)
            if row["state"] == "excluded":
                raise SourceExcluded(row["reason"])
            if row["state"] not in {"screened", "prepared"}:
                raise ValueError("preparation requires an explicit completed selection")
            receipt_path = self.layout.path(
                "sources", source_id, self.plan.preparation.digest, "prepared.json"
            )
            if row["preparation"] is not None:
                reference = ArtifactRef.model_validate_json(row["preparation"])
            elif row["state"] == "prepared":
                raise ValueError(
                    "prepared source has no registered preparation receipt"
                )
            elif receipt_path.exists():
                reference = annotation_reference(self.layout, receipt_path)
            else:
                acquired = read_document(
                    self.layout,
                    ArtifactRef.model_validate_json(row["acquisition"]),
                    AcquisitionReceipt,
                )
                selection = ArtifactRef.model_validate_json(row["selection"])
                reference = prepare_source(
                    self.layout,
                    acquired.source,
                    self.plan.preparation,
                    self.config,
                    selection,
                )
            # Validate the fixed receipt and every artifact before reconciling the
            # ledger. Published work must never enter the generation path again.
            if not self.ledger.record_preparation(self.plan, reference):
                raise SourceExcluded(self.ledger.source(self.plan, source_id)["reason"])
            return reference
