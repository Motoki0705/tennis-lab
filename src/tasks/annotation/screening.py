"""Connect source selection to the common worker and actual rollout evidence."""

from __future__ import annotations

from pathlib import Path

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import SelectionReceipt
from src.tasks.base.annotation.collection.service import Collection
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan
from src.tasks.base.annotation.screening.contracts import (
    PresentedImages,
    PreviewReceipt,
    ScreeningRequest,
)
from src.tasks.base.annotation.screening.service import Screening
from src.tasks.base.annotation.telemetry.capture import presented_images
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    PreparedWorker,
    WorkOutcome,
    WorkRequest,
)


class ScreeningAdapter:
    name = "source_screening"
    version = "v1"

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
            prompt=annotation_reference(layout, directory / "prompt.md")
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
        presentation = write_document(
            layout,
            directory / "presented.json",
            PresentedImages(
                request_digest=ref.digest,
                events=telemetry,
                images=presented_images(layout, telemetry),
            ),
        )
        selection_ref = Screening(Collection(layout, plan)).finalize(
            ref, result, presentation, self.previews(layout, directory)
        )
        selection = read_document(layout, selection_ref, SelectionReceipt)
        return WorkOutcome(
            status="complete" if selection.windows else "not_applicable",
            receipt=selection_ref,
            reason=None if selection.windows else selection.shortage_reason,
        )
