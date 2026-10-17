"""Bind human/agent clip choices to frozen inputs and actual presentation evidence."""

from __future__ import annotations

import json
import shlex
from fractions import Fraction
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

from ..artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from ..collection.contracts import AcquisitionReceipt, SelectionReceipt, TimeWindow
from ..collection.service import Collection, SourceExcluded
from ..contracts import ArtifactRef, Document
from ..media import probe_video
from ..paths import AnnotationLayout
from ..plan import references
from .contracts import (
    ClipDecision,
    PresentedImages,
    PreviewReceipt,
    ScreeningEvidence,
    ScreeningInput,
    ScreeningPolicy,
    ScreeningRequest,
    SourceDecision,
    check_decision,
)
from .images import make_sheets


class PreflightExclusion(Document):
    plan_digest: str
    acquisition: ArtifactRef
    policy: ArtifactRef
    reason: str


def attempt_directory(layout: AnnotationLayout, request: ScreeningRequest) -> Path:
    path: Path = layout.artifact_path(
        ArtifactRef(
            role="annotation", path=request.attempt_path, digest="0" * 64, size_bytes=0
        )
    )
    return path


def request_preview(
    layout: AnnotationLayout,
    request: ScreeningRequest,
    *,
    start: float,
    stop: float,
    step: float,
) -> ArtifactRef:
    if stop > request.source_duration:
        raise ValueError("preview interval extends beyond the source")
    return make_sheets(
        layout,
        request.source.media,
        attempt_directory(layout, request) / "previews",
        origin_pts=request.source_origin_pts,
        time_base=request.time_base,
        start=start,
        stop=stop,
        step=step,
    )


def _covers_detail_window(
    preview: PreviewReceipt, clip: ClipDecision, maximum_gap: float
) -> bool:
    """Measure actual displayed samples, including both selected-window edges."""
    start, stop = Fraction(str(clip.start)), Fraction(str(clip.stop))
    clock = Fraction(preview.time_base)
    timestamps = (
        (sample.source_pts - preview.origin_pts) * clock
        for page in preview.pages
        for sample in page.samples
    )
    samples = [timestamp for timestamp in timestamps if start <= timestamp < stop]
    if not samples:
        return False
    boundaries = [start, *samples, stop]
    limit = Fraction(str(maximum_gap))
    return all(
        later - earlier <= limit
        for earlier, later in zip(boundaries, boundaries[1:], strict=False)
    )


class Screening:
    """No model launch or second image review; orchestration supplies tool evidence."""

    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        self.layout, self.plan = collection.layout, collection.plan
        self.policy = read_document(self.layout, self.plan.selection, ScreeningPolicy)

    def prepare(self, source_id: str) -> ArtifactRef:
        """CPU stage: probe once and render the shared, read-only source overview."""
        directory = self.layout.path(
            "campaigns", self.plan.campaign_id, "screening", source_id
        )
        row = self.collection.ledger.source(self.plan, source_id)
        if row["state"] not in {"acquired", "screened", "prepared"}:
            raise ValueError("screening requires an acquired, nonexcluded source")
        acquisition = ArtifactRef.model_validate_json(row["acquisition"])
        acquired = read_document(self.layout, acquisition, AcquisitionReceipt)
        prepared_path = directory / "input.json"
        if prepared_path.exists():
            prior_ref = annotation_reference(self.layout, prepared_path)
            prior = read_document(self.layout, prior_ref, ScreeningInput)
            if (prior.plan_digest, prior.source, prior.policy) != (
                self.plan.sha256,
                acquired.source,
                self.plan.selection,
            ):
                raise ValueError(
                    "screening input differs from the registered source/plan"
                )
            for ref in references(prior):
                verify_reference(self.layout, ref)
            return prior_ref
        source_path = verify_reference(self.layout, acquired.source.media)
        try:
            timeline = probe_video(source_path)
            duration = timeline.boundary_seconds(len(timeline.pts))
            if (
                not self.policy.window_seconds
                <= duration
                <= self.policy.maximum_source_seconds
            ):
                raise ValueError(
                    "source duration is outside the declared screening limits"
                )
        except ValueError as error:
            reason = f"screening preflight: {error}"
            evidence = write_document(
                self.layout,
                directory / "exclusion.json",
                PreflightExclusion(
                    plan_digest=self.plan.sha256,
                    acquisition=acquisition,
                    policy=self.plan.selection,
                    reason=reason,
                ),
            )
            selection = write_document(
                self.layout,
                directory / "selection.json",
                SelectionReceipt(
                    source_id=source_id,
                    source_digest=acquired.source.media.digest,
                    policy=self.plan.selection,
                    evidence=evidence,
                    requested_maximum=self.plan.counts.max_clips_per_video,
                    windows=(),
                    shortage_reason=reason,
                ),
            )
            self.collection.select(selection)
            raise SourceExcluded(reason) from error
        overview = make_sheets(
            self.layout,
            acquired.source.media,
            directory / "previews",
            origin_pts=timeline.pts[0],
            time_base=str(timeline.time_base),
            start=0.0,
            stop=duration,
            step=self.policy.overview_step_seconds,
        )
        return write_document(
            self.layout,
            prepared_path,
            ScreeningInput(
                plan_digest=self.plan.sha256,
                source=acquired.source,
                policy=self.plan.selection,
                source_duration=duration,
                source_origin_pts=timeline.pts[0],
                time_base=str(timeline.time_base),
                overview=overview,
            ),
        )

    def request(self, prepared: ArtifactRef, directory: Path) -> ArtifactRef:
        """Only bind small inputs while the worker admission lock is held."""
        source_input = read_document(self.layout, prepared, ScreeningInput)
        expected = self.layout.path("campaigns", self.plan.campaign_id, "workers")
        if (
            not directory.is_relative_to(expected)
            or len(directory.relative_to(expected).parts) != 2
        ):
            raise ValueError("screening request must belong to a worker attempt")
        if (source_input.plan_digest, source_input.policy) != (
            self.plan.sha256,
            self.plan.selection,
        ):
            raise ValueError("screening input belongs to another plan")
        request = ScreeningRequest(
            **source_input.model_dump(),
            prepared_input=prepared,
            requested_maximum=self.plan.counts.max_clips_per_video,
            attempt_path=directory.relative_to(self.layout.root).as_posix(),
        )
        reference = write_document(self.layout, directory / "request.json", request)
        write_asset(
            self.layout,
            directory / "decision.schema.json",
            (
                json.dumps(
                    SourceDecision.model_json_schema(), ensure_ascii=False, indent=2
                )
                + "\n"
            ).encode(),
        )
        instructions = Path(__file__).with_name("WORKER.md").read_text()
        code_root = PROJECT_ROOT
        command = shlex.join(
            [
                "env",
                f"PYTHONPATH={code_root}",
                str(code_root / ".venv/bin/python"),
                "-m",
                "src.tasks.base.annotation.screening",
                "--request",
                str(self.layout.artifact_path(reference)),
                "--request-sha256",
                reference.digest,
            ]
        )
        prompt = (
            f"{instructions}\n\nImmutable request: {self.layout.artifact_path(reference)}\n"
            f"Request SHA-256: {reference.digest}\n"
            f"Overview receipt: {self.layout.artifact_path(request.overview)}\n"
            f"Decision schema: {directory / 'decision.schema.json'}\n"
            f"Write the decision to {directory / 'work/result.json'}.\n"
            f"Registered tool command: {command}\n"
        )
        write_asset(self.layout, directory / "prompt.md", prompt.encode())
        (directory / "work").mkdir(exist_ok=True)
        return reference

    def finalize(
        self,
        request_ref: ArtifactRef,
        decision_ref: ArtifactRef,
        presentation_ref: ArtifactRef,
        preview_refs: tuple[ArtifactRef, ...],
    ) -> ArtifactRef:
        request = read_document(self.layout, request_ref, ScreeningRequest)
        if (request.plan_digest, request.policy) != (
            self.plan.sha256,
            self.plan.selection,
        ):
            raise ValueError("screening request does not match the frozen campaign")
        row = self.collection.ledger.source(self.plan, request.source.canonical_id)
        acquired = read_document(
            self.layout,
            ArtifactRef.model_validate_json(row["acquisition"]),
            AcquisitionReceipt,
        )
        if request.source != acquired.source:
            raise ValueError("screening request source differs from the ledger")
        directory = attempt_directory(self.layout, request)
        expected = self.layout.path("campaigns", self.plan.campaign_id, "workers")
        if (
            not directory.is_relative_to(expected)
            or len(directory.relative_to(expected).parts) != 2
            or self.layout.artifact_path(request_ref) != directory / "request.json"
        ):
            raise ValueError("screening request is outside its declared attempt")
        original = read_document(self.layout, request.prepared_input, ScreeningInput)
        if any(
            getattr(request, key) != getattr(original, key)
            for key in ScreeningInput.model_fields
        ):
            raise ValueError("screening request changed its prepared source/overview")
        verify_reference(self.layout, request.source.media)
        decision = read_document(self.layout, decision_ref, SourceDecision)
        check_decision(request, self.policy, decision)
        presentation = read_document(self.layout, presentation_ref, PresentedImages)
        if presentation.request_digest != request_ref.digest:
            raise ValueError("presentation evidence belongs to another request")
        verify_reference(self.layout, presentation.events)
        presented = set(presentation.images)
        for image in presented:
            verify_reference(self.layout, image)
        refs = tuple(dict.fromkeys((request.overview, *preview_refs)))
        previews: list[PreviewReceipt] = []
        images: dict[str, ArtifactRef] = {}
        for reference in refs:
            path = self.layout.artifact_path(reference)
            if reference != request.overview and not path.is_relative_to(
                directory / "previews"
            ):
                raise ValueError("preview evidence is outside the screening attempt")
            preview = read_document(self.layout, reference, PreviewReceipt)
            if (preview.source, preview.origin_pts, preview.time_base) != (
                request.source.media,
                request.source_origin_pts,
                request.time_base,
            ):
                raise ValueError("preview evidence belongs to another source/clock")
            if (
                preview.stop > request.source_duration
                or preview.decoded_until + 1e-6 < preview.stop
            ):
                raise ValueError("preview did not decode the declared interval")
            for page in preview.pages:
                if self.layout.artifact_path(page.image).parent != path.parent:
                    raise ValueError(
                        "preview image is outside its immutable page directory"
                    )
                verify_reference(self.layout, page.image)
                images[page.image.path] = page.image
            previews.append(preview)
        overview = previews[0]
        if (overview.start, overview.stop, overview.step) != (
            0.0,
            request.source_duration,
            self.policy.overview_step_seconds,
        ):
            raise ValueError("overview does not cover the source at the declared step")
        if any(page.image not in presented for page in overview.pages):
            raise ValueError("all overview pages must actually be presented")
        for clip in decision.clips:
            cited: set[ArtifactRef] = set()
            for image_path in clip.evidence_images:
                if image_path not in images or images[image_path] not in presented:
                    raise ValueError(
                        "selected clip cites an unknown or unpresented image"
                    )
                cited.add(images[image_path])
            detailed = [
                preview
                for preview in previews
                if preview.start <= clip.start
                and preview.stop >= clip.stop
                and preview.step <= self.policy.detail_step_seconds
                and all(page.image in presented for page in preview.pages)
                and any(page.image in cited for page in preview.pages)
            ]
            if not detailed:
                raise ValueError(
                    "selected clip requires presented detailed coverage of the whole window"
                )
            if not any(
                _covers_detail_window(preview, clip, self.policy.detail_step_seconds)
                for preview in detailed
            ):
                raise ValueError(
                    "selected clip requires actual PTS sample gaps no greater than "
                    f"{self.policy.detail_step_seconds} seconds, including window boundaries"
                )
        decision_ref = write_asset(
            self.layout,
            directory / "decision.json",
            verify_reference(self.layout, decision_ref).read_bytes(),
        )
        evidence = write_document(
            self.layout,
            directory / "evidence.json",
            ScreeningEvidence(
                request=request_ref,
                decision=decision_ref,
                presentation=presentation_ref,
                previews=refs,
            ),
        )
        selection = write_document(
            self.layout,
            directory / "selection.json",
            SelectionReceipt(
                source_id=request.source.canonical_id,
                source_digest=request.source.media.digest,
                policy=request.policy,
                evidence=evidence,
                requested_maximum=request.requested_maximum,
                windows=tuple(
                    TimeWindow(
                        start=clip.start,
                        stop=clip.stop,
                        target_start=clip.start + self.policy.context_seconds,
                        target_stop=clip.stop - self.policy.context_seconds,
                    )
                    for clip in decision.clips
                ),
                shortage_reason=decision.notes
                if decision.status == "insufficient"
                else None,
            ),
        )
        self.collection.select(selection)
        return selection
