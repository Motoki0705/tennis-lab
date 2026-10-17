"""Ball worker protocol: native images, bounded edits, retry evidence and final adoption."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Literal

from pydantic import Field

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
    Document,
    FrameRange,
)
from src.tasks.base.annotation.editing import ReviewContext, Revision, RevisionStore
from src.tasks.base.annotation.handoff import (
    HandoffContent,
    build_handoff,
    read_handoff,
)
from src.tasks.base.annotation.images.contracts import Point, RenderRequest, Tile
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, references
from src.tasks.base.annotation.predictions import PredictionReceipt
from src.tasks.base.annotation.storage.references import distinct
from src.tasks.base.annotation.telemetry.capture import verify_report
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    PreparedWorker,
    WorkOutcome,
    WorkRequest,
)
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from src.tasks.base.annotation.workers.review.contracts import (
    ImageRequest,
    ReviewWorkerInput,
    ReviewWorkerResult,
    RowsRequest,
)
from src.tasks.base.annotation.workers.review.service import (
    image_catalog,
    resolve_attempt,
)
from src.utils.paths import PROJECT_ROOT

from .adapter import BallReviewAdapter
from .contracts import BallFrame, BallLabels
from .evidence import review_images
from .storage import retained_revision
from .validation import inputs

ProofState = Literal["unreviewed", "verified", "pending_current_attempt", "missing"]


def proof_states(
    layout: AnnotationLayout,
    context: ReviewContext,
    labels: BallLabels,
) -> tuple[dict[int, ProofState], dict[ArtifactRef, str]]:
    plan = read_document(layout, context.plan, CampaignPlan)
    context_ref = annotation_reference(
        layout, layout.path("annotations", "ball", labels.clip_id, "context.json")
    )
    work = WorkRequest(
        adapter="ball_review",
        adapter_version="v1",
        stage="review",
        subject_id=labels.clip_id,
        payload=context_ref,
        inputs=(context.manifest, context.prediction),
    )
    ledger = WorkerLedger(layout)
    history = ledger.history(plan, work)
    current = ledger.task(work.work_id)["current_attempt"]
    proven, pending = {}, set()
    unavailable: dict[ArtifactRef, str] = {}
    for attempt in history:
        if attempt.telemetry is not None:
            proven[attempt.launch], reason = review_images(layout, attempt.telemetry)
            if reason is not None:
                unavailable[attempt.launch] = reason
        elif attempt.state == "running" and attempt.execution is None:
            from src.tasks.base.annotation.workers.contracts import Launch

            if read_document(layout, attempt.launch, Launch).attempt == current:
                pending.add(attempt.launch)
    catalog = {image.sha256: image.image for image in labels.images}
    result: dict[int, ProofState] = {}
    for frame in labels.frames:
        if not frame.decision.reviewed:
            state: ProofState = "unreviewed"
        elif all(
            catalog[key] in proven.get(frame.review_actor, set())
            for key in frame.decision.evidence
        ):
            state = "verified"
        else:
            state = (
                "pending_current_attempt"
                if frame.review_actor in pending
                else "missing"
            )
        result[frame.frame.frame_index] = state
    return result, unavailable


class BallRowPage(Document):
    revision: ArtifactRef
    scope: FrameRange
    rows: tuple[BallFrame, ...] = Field(min_length=1, max_length=8)
    image_proof: tuple[ProofState, ...] = Field(min_length=1, max_length=8)
    image_proof_reasons: tuple[str | None, ...] = Field(min_length=1, max_length=8)
    images: tuple[RenderedImage, ...] = Field(max_length=256)


class UnusableImageEvidence(Document):
    actor: ArtifactRef
    reason: str


class UnusableImageEvidencePage(Document):
    payload: ArtifactRef
    entries: tuple[UnusableImageEvidence, ...] = Field(min_length=1, max_length=8)
    next_page: ArtifactRef | None


def _evidence_pages(
    layout: AnnotationLayout,
    clip_id: str,
    payload: ArtifactRef,
    unavailable: dict[ArtifactRef, str],
) -> ArtifactRef | None:
    entries = tuple(
        UnusableImageEvidence(actor=actor, reason=reason)
        for actor, reason in unavailable.items()
    )
    next_page: ArtifactRef | None = None
    for start in reversed(range(0, len(entries), 8)):
        page = UnusableImageEvidencePage(
            payload=payload, entries=entries[start : start + 8], next_page=next_page
        )
        next_page = write_document(
            layout,
            layout.path("annotations", "ball", clip_id, "image-proof", page.sha256 + ".json"),
            page,
        )
    return next_page


class BallReviewServices(BallReviewAdapter):
    def summarize(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        payload: ArtifactRef,
        scope: FrameRange,
        after_row: str | None,
    ) -> HandoffContent:
        summary = super().summarize(
            layout, context, manifest, payload, scope, after_row
        )
        states, unavailable = proof_states(
            layout, context, read_document(layout, payload, BallLabels)
        )
        missing = any(states[row.frame_index] == "missing" for row in summary.rows)
        evidence_page = _evidence_pages(layout, manifest.clip_id, payload, unavailable)
        evidence_actions = (
            (
                f"Unusable image evidence: {len(unavailable)} attempts. "
                "Reconfirm affected decisions. Read reasons and follow next_page: "
                + evidence_page.model_dump_json(),
            )
            if evidence_page is not None
            else ()
        )
        return summary.model_copy(
            update={
                "rows": tuple(
                    row.model_copy(
                        update={
                            "detail": row.detail
                            + " image_proof="
                            + states[row.frame_index]
                        }
                    )
                    for row in summary.rows
                ),
                "next_actions": (
                    "Reinspect rows with image_proof=missing and explicitly replace their decisions; retain unresolved notes until resolved.",
                    *evidence_actions,
                    *summary.next_actions,
                )
                if missing
                else (*evidence_actions, *summary.next_actions),
            }
        )

    def rows(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        request: RowsRequest,
    ) -> BallRowPage:
        node = read_document(layout, request.expected_revision, Revision)
        if read_document(layout, node.context, ReviewContext) != context:
            raise ValueError("row request belongs to another clip context")
        labels = read_document(layout, node.payload, BallLabels)
        states, unavailable = proof_states(layout, context, labels)
        rows = labels.frames[request.scope.start : request.scope.stop]
        used = {key for frame in rows for key in frame.decision.evidence}
        return BallRowPage(
            revision=request.expected_revision,
            scope=request.scope,
            rows=rows,
            image_proof=tuple(states[row.frame.frame_index] for row in rows),
            image_proof_reasons=tuple(
                unavailable.get(row.review_actor) if row.review_actor else None
                for row in rows
            ),
            images=tuple(image for image in labels.images if image.sha256 in used),
        )

    def render_request(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        request: ImageRequest,
    ) -> RenderRequest:
        _, _, predictions = inputs(layout, context, manifest)
        labels = read_document(
            layout,
            read_document(layout, request.expected_revision, Revision).payload,
            BallLabels,
        )
        if any(t.frame_index >= len(manifest.frames) for t in request.tiles):
            raise ValueError("requested image frame lies outside the clip")
        tiles = []
        for tile in request.tiles:
            markers: tuple[Point, ...]
            if request.drawing == "prediction":
                row = predictions.rows[tile.frame_index]
                markers = (Point(x=row.x, y=row.y),)
            elif request.drawing == "revision":
                markers = tuple(
                    Point(x=o.center[0], y=o.center[1], color="cyan")
                    for o in labels.frames[tile.frame_index].decision.observations
                    if o.center is not None
                )
            else:
                markers = ()
            tiles.append(
                Tile(frame_index=tile.frame_index, crop=tile.crop, drawings=markers)
            )
        prediction = request.drawing == "prediction"
        return RenderRequest(
            manifest=context.manifest,
            annotation=context.prediction if prediction else request.expected_revision,
            annotation_kind="prediction" if prediction else "revision",
            drawing=request.drawing,
            reason=request.reason,
            tiles=tuple(tiles),
            columns=request.columns,
            tile_width=request.tile_width,
            tile_height=request.tile_height,
        )


class BallReviewWorker:
    name = "ball_review"
    version = "v1"

    def input_references(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
    ) -> tuple[ArtifactRef, ...]:
        self.require_work(request)
        context = read_document(layout, request.payload, ReviewContext)
        manifest = read_document(layout, context.manifest, ClipManifest)
        receipt = read_document(layout, context.prediction, PredictionReceipt)
        if (
            read_document(layout, context.plan, CampaignPlan) != plan
            or context.task != "ball"
            or manifest.clip_id != request.subject_id
            or request.inputs != (context.manifest, context.prediction)
            or receipt.context.manifest != context.manifest
            or receipt.context.clip_id != manifest.clip_id
        ):
            raise ValueError(
                "ball input references differ from the fixed campaign/clip"
            )
        result: tuple[ArtifactRef, ...] = distinct(
            layout,
            (*references(context), *references(manifest), receipt.prediction.artifact),
        )
        return result

    @classmethod
    def require_work(cls, request: WorkRequest) -> None:
        if (request.adapter, request.adapter_version, request.stage) != (
            cls.name,
            cls.version,
            "review",
        ):
            raise ValueError(
                "ball review worker requires its exact adapter/version/stage"
            )

    def prepare(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> PreparedWorker:
        self.require_work(request)
        context = read_document(layout, request.payload, ReviewContext)
        adapter, store = BallReviewServices(), RevisionStore(layout)
        bound_plan, manifest, _ = store.context(context, adapter)
        if (
            bound_plan != plan
            or request.subject_id != manifest.clip_id
            or request.inputs != (context.manifest, context.prediction)
        ):
            raise ValueError("ball worker input differs from its fixed campaign/clip")
        revision = store.initialize(context, adapter)
        if read_document(layout, revision, Revision).context != request.payload:
            raise ValueError("worker must use the registered review context reference")
        handoff = build_handoff(
            store,
            request.payload,
            adapter,
            FrameRange(
                start=manifest.target_range.start,
                stop=min(manifest.target_range.stop, manifest.target_range.start + 64),
            ),
        )
        body = ReviewWorkerInput(
            work=request,
            attempt_path=directory.relative_to(layout.root).as_posix(),
            initial_handoff=handoff,
        )
        ref = write_document(layout, directory / "request.json", body)
        command = shlex.join(
            [
                str(PROJECT_ROOT / ".venv/bin/python"),
                "-m",
                "src.tasks.base.annotation.workers.review",
                "--request",
                str(layout.artifact_path(ref)),
                "--request-sha256",
                ref.digest,
            ]
        )
        _, policy, _ = inputs(layout, context, manifest)
        instructions = Path(__file__).with_name("worker-instructions.md").read_text()
        prompt = (
            instructions
            + "\n\nFixed invocation (set PYTHONPATH to "
            + shlex.quote(str(PROJECT_ROOT))
            + "):\n"
            + command
            + "\n\nNative geometry and target interval:\n"
            + json.dumps(
                {
                    "width": manifest.width,
                    "height": manifest.height,
                    "frames": len(manifest.frames),
                    "time_base": manifest.time_base,
                    "target": manifest.target_range.model_dump(),
                }
            )
            + "\n\nFrozen review policy:\n"
            + policy.model_dump_json()
            + "\n\nInitial bounded handoff:\n"
            + read_handoff(layout, handoff).model_dump_json()
            + "\n\nAnnotation artifact root (read-only):\n"
            + str(layout.root)
        )
        return PreparedWorker(
            prompt=write_asset(layout, directory / "prompt.md", prompt.encode()),
            inputs=distinct(
                layout, (ref, handoff, *retained_revision(layout, context, revision))
            ),
        )

    def telemetry_images(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> tuple[RenderedImage, ...]:
        self.require_work(request)
        attempt = resolve_attempt(
            layout,
            annotation_reference(layout, directory / "request.json"),
            plan=plan,
            work=request,
        )
        head, _ = RevisionStore(layout).current(request.payload, BallReviewServices())
        labels = read_document(
            layout, read_document(layout, head, Revision).payload, BallLabels
        )
        # A retry can view an already-cited image again. Only actual rollout bytes
        # count; merely including an old image here provides no review evidence.
        images = {
            image.sha256: image
            for image in (*labels.images, *image_catalog(layout, attempt))
        }
        return tuple(images.values())

    def classify_tool(self, name: str | None, raw_input: str) -> str:
        if name and "view_image" in name:
            return "ball_review_image_presentation"
        for operation in (
            "review_image",
            "review_edit",
            "review_handoff",
            "review_rows",
        ):
            if (
                "src.tasks.base.annotation.workers.review" in raw_input
                and operation in raw_input
            ):
                return operation
        return "ball_review_inspection"

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
        self.require_work(request)
        attempt = resolve_attempt(
            layout,
            annotation_reference(layout, directory / "request.json"),
            plan=plan,
            work=request,
        )
        if (
            telemetry is None
            or execution.status != "succeeded"
            or execution.exit_code != 0
            or execution.launch_digest != attempt.actor.digest
        ):
            raise ValueError(
                "ball review completion requires its successful worker and actual rollout evidence"
            )
        report = verify_report(layout, telemetry)
        if (
            report.launch != attempt.actor
            or read_document(layout, report.execution, ExitReceipt) != execution
        ):
            raise ValueError("ball completion telemetry belongs to another attempt")
        decision = read_document(layout, result, ReviewWorkerResult)
        if not decision.reason.strip():
            raise ValueError("review completion/deferral requires a nonempty reason")
        store, adapter = RevisionStore(layout), BallReviewServices()
        head, sealed = store.current(request.payload, adapter)
        if decision.revision != head:
            raise ValueError(
                "worker completion does not name the current review revision"
            )
        if decision.status == "deferred":
            if sealed is not None:
                raise ValueError("an adopted review cannot be deferred")
            return WorkOutcome(
                status="deferred", receipt=result, reason=decision.reason
            )
        context = read_document(layout, request.payload, ReviewContext)
        manifest = read_document(layout, context.manifest, ClipManifest)
        history = WorkerLedger(layout).history(plan, request)
        reports = tuple(
            item.telemetry for item in history if item.telemetry is not None
        )
        if not any(
            item.launch == attempt.actor and item.telemetry == telemetry
            for item in history
        ):
            raise ValueError(
                "current rollout is missing from registered attempt history"
            )
        if sealed is None:
            adoption = adapter.prepare_adoption(
                layout, context, manifest, head, reports, reason=decision.reason
            )
        else:
            adoption = sealed
            adapter.validate_adoption(
                layout,
                context,
                manifest,
                head,
                read_document(layout, sealed, AdoptionReceipt),
            )
        store.seal(head, adoption, adapter)
        return WorkOutcome(status="complete", receipt=result, adoption=adoption)
