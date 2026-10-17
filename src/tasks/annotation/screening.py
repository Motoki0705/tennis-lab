"""Connect source selection to the common worker and actual rollout evidence."""

from __future__ import annotations

from pathlib import Path

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import SelectionReceipt
from src.tasks.base.annotation.collection.service import Collection
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, references
from src.tasks.base.annotation.screening.contracts import (
    PresentedImages,
    PreviewReceipt,
    ScreeningInput,
    ScreeningRequest,
    SourceDecision,
    check_decision,
)
from src.tasks.base.annotation.screening.service import Screening
from src.tasks.base.annotation.storage.references import distinct
from src.tasks.base.annotation.telemetry.capture import presented_images
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    PreparedWorker,
    WorkOutcome,
    WorkRequest,
)
from src.tasks.base.annotation.workers.services import ServiceReply, ServiceRequest


def unfinished_services(
    layout: AnnotationLayout, directory: Path, task_request: ArtifactRef
) -> tuple[str, ...]:
    """Do not turn pending or failed image requests into a material shortage."""
    reasons: list[str] = []
    for path in sorted((directory / "work/service-requests").glob("*.json")):
        if path.stat().st_size > 32768:
            raise ValueError("worker service request exceeds 32 KiB")
        request = read_document(
            layout, annotation_reference(layout, path), ServiceRequest
        )
        if request.task_request != task_request or path.stem != request.sha256:
            raise ValueError("screening service changed its task/attempt identity")
        reply_path = directory / "services" / request.sha256 / "reply.json"
        if not reply_path.exists():
            reasons.append(f"{request.sha256}: pending")
            continue
        reply = read_document(
            layout, annotation_reference(layout, reply_path), ServiceReply
        )
        if reply.request_digest != request.sha256:
            raise ValueError("screening service response belongs to another request")
        if reply.status != "succeeded" or reply.result is None:
            reasons.append(f"{request.sha256}: failed ({reply.reason})")
        else:
            verify_reference(layout, reply.result)
    return tuple(reasons)


class ScreeningAdapter:
    name = "source_screening"
    version = "v1"

    def input_references(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
    ) -> tuple[ArtifactRef, ...]:
        body = read_document(layout, request.payload, ScreeningInput)
        overview = read_document(layout, body.overview, PreviewReceipt)
        if (
            body.plan_digest != plan.sha256
            or body.source.canonical_id != request.subject_id
            or body.policy != plan.selection
            or overview.source != body.source.media
        ):
            raise ValueError(
                "screening input references differ from the fixed source/plan"
            )
        result: tuple[ArtifactRef, ...] = distinct(
            layout, (*references(body), *references(overview))
        )
        return result

    def prepare(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> PreparedWorker:
        screening = Screening(Collection(layout, plan))
        ref = screening.request(request.payload, directory)
        body = read_document(layout, ref, ScreeningRequest)
        if body.source.canonical_id != request.subject_id:
            raise ValueError("screening work identity differs from the prepared source")
        return PreparedWorker(
            prompt=annotation_reference(layout, directory / "prompt.md"),
            inputs=(
                ref,
                annotation_reference(layout, directory / "decision.schema.json"),
            ),
        )

    @staticmethod
    def previews(layout: AnnotationLayout, directory: Path) -> tuple[ArtifactRef, ...]:
        return tuple(
            annotation_reference(layout, path)
            for path in sorted((directory / "previews").glob("*/preview.json"))
        )

    def telemetry_images(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> tuple[RenderedImage, ...]:
        del plan, request
        body = read_document(
            layout,
            annotation_reference(layout, directory / "request.json"),
            ScreeningRequest,
        )
        result: list[RenderedImage] = []
        for ref in (body.overview, *self.previews(layout, directory)):
            preview = read_document(layout, ref, PreviewReceipt)
            result.extend(
                RenderedImage(image=page.image, metadata=ref) for page in preview.pages
            )
        return tuple(result)

    def classify_tool(self, name: str | None, raw_input: str) -> str:
        del name
        if "src.tasks.base.annotation.screening" in raw_input:
            return (
                "screening_image_request"
                if "images" in raw_input
                else "screening_structural_check"
            )
        return "screening_inspection"

    def validate(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
        result: ArtifactRef,
        execution: ExitReceipt,
        telemetry: ArtifactRef | None,
    ) -> WorkOutcome:
        if telemetry is None or execution.status != "succeeded":
            raise ValueError(
                "screening requires actual identified rollout image evidence"
            )
        ref = annotation_reference(layout, directory / "request.json")
        body = read_document(layout, ref, ScreeningRequest)
        if (
            body.prepared_input != request.payload
            or body.source.canonical_id != request.subject_id
        ):
            raise ValueError("worker changed its source/input binding")
        screening = Screening(Collection(layout, plan))
        decision = read_document(layout, result, SourceDecision)
        check_decision(body, screening.policy, decision)
        pending = unfinished_services(layout, directory, ref)
        if decision.status == "deferred" or pending:
            reason = "screening is technically incomplete: " + decision.notes
            if pending:
                reason += "; unresolved image services: " + "; ".join(pending)
            return WorkOutcome(status="deferred", receipt=result, reason=reason)
        presentation = write_document(
            layout,
            directory / "presented.json",
            PresentedImages(
                request_digest=ref.digest,
                events=telemetry,
                images=presented_images(layout, telemetry),
            ),
        )
        selection_ref = screening.finalize(
            ref, result, presentation, self.previews(layout, directory)
        )
        selection = read_document(layout, selection_ref, SelectionReceipt)
        return WorkOutcome(
            status="complete" if selection.windows else "not_applicable",
            receipt=selection_ref,
            reason=None if selection.windows else selection.shortage_reason,
        )
