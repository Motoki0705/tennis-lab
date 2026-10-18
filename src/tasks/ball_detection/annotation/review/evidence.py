"""Connect each visual claim to native crops and the worker that saw their bytes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    ReviewEvidence,
)
from src.tasks.base.annotation.editing import ReviewContext, Revision
from src.tasks.base.annotation.images import read_rendered
from src.tasks.base.annotation.images.contracts import (
    Point,
    RenderReceipt,
    RenderRequest,
    TileReceipt,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.telemetry.capture import (
    ImagePresentationUnavailable,
    presented_images,
    verify_report,
)
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import Launch

from ..contracts import PredictionDocument
from .contracts import BallFrame, BallLabels, BallReviewPolicy, BallReviewProof
from .validation import ranges, resolved


def review_images(
    layout: AnnotationLayout, telemetry: ArtifactRef
) -> tuple[set[ArtifactRef], str | None]:
    """Keep unusable attempts for audit, but never use them as visual proof."""
    try:
        return set(presented_images(layout, telemetry)), None
    except ImagePresentationUnavailable as error:
        # Integrity failures are different exceptions and must still stop review.
        return set(), str(error)


def lineage(
    layout: AnnotationLayout, context: ReviewContext, revision: ArtifactRef
) -> set[ArtifactRef]:
    result: set[ArtifactRef] = set()
    previous_sequence: int | None = None
    current: ArtifactRef | None = revision
    while current is not None:
        node = read_document(layout, current, Revision)
        if (
            current in result
            or (
                previous_sequence is not None and node.sequence != previous_sequence - 1
            )
            or read_document(layout, node.context, ReviewContext) != context
        ):
            raise ValueError("image revision is not in a consistent review lineage")
        result.add(current)
        previous_sequence, current = node.sequence, node.previous
    return result


@dataclass(frozen=True)
class Catalog:
    images: dict[str, RenderedImage]
    receipts: dict[str, RenderReceipt]

    def frames(self, evidence: tuple[str, ...], index: int) -> tuple[str, ...]:
        if len(set(evidence)) != len(evidence) or any(
            key not in self.images for key in evidence
        ):
            raise ValueError("unknown or repeated review image identity")
        return tuple(
            key
            for key in evidence
            if any(tile.frame_index == index for tile in self.receipts[key].tiles)
        )


def catalog(
    layout: AnnotationLayout,
    context: ReviewContext,
    images: tuple[RenderedImage, ...],
    predictions: PredictionDocument,
    *,
    allowed_revisions: set[ArtifactRef] | None = None,
) -> Catalog:
    saved: dict[str, RenderedImage] = {}
    receipts: dict[str, RenderReceipt] = {}
    for image in images:
        if image.sha256 in saved:
            raise ValueError("review image catalog contains duplicates")
        request, receipt = read_rendered(layout, image)
        if request.manifest != context.manifest:
            raise ValueError("review image belongs to another clip")
        labels: BallLabels | None = None
        if request.annotation_kind == "prediction":
            if request.annotation != context.prediction:
                raise ValueError("review image belongs to another prediction")
        else:
            node = read_document(layout, request.annotation, Revision)
            if read_document(layout, node.context, ReviewContext) != context or (
                allowed_revisions is not None
                and request.annotation not in allowed_revisions
            ):
                raise ValueError("review image is from a different or future revision")
            labels = read_document(layout, node.payload, BallLabels)
        _drawings(request, predictions, labels)
        saved[image.sha256], receipts[image.sha256] = image, receipt
    return Catalog(saved, receipts)


def _drawings(
    request: RenderRequest, predictions: PredictionDocument, labels: BallLabels | None
) -> None:
    for tile in request.tiles:
        index = tile.frame_index
        if labels is None:
            center = predictions.rows[index].center
            expected = {center} if center is not None else set()
        else:
            expected = {
                o.center
                for o in labels.frames[index].decision.observations
                if o.center is not None
            }
        if any(
            not isinstance(item, Point) or (item.x, item.y) not in expected
            for item in tile.drawings
        ):
            raise ValueError(
                "ball overlay markers differ from their declared annotation"
            )


def _large_enough(tile: TileReceipt, minimum: float) -> bool:
    return min(Fraction(tile.scale_x), Fraction(tile.scale_y)) >= Fraction(str(minimum))


def check_frame_images(
    frame: BallFrame, manifest: ClipManifest, policy: BallReviewPolicy, saved: Catalog
) -> None:
    decision = frame.decision
    if not decision.reviewed:
        return
    indices = saved.frames(decision.evidence, frame.frame.frame_index)
    if indices != decision.evidence:
        raise ValueError("each cited image must contain the claimed frame")
    tiles = [
        tile
        for key in indices
        for tile in saved.receipts[key].tiles
        if tile.frame_index == frame.frame.frame_index
        and _large_enough(tile, policy.minimum_image_scale)
    ]
    if not tiles:
        raise ValueError("reviewed frame lacks a sufficiently detailed native image")
    absent = decision.presence == "absent" or any(
        o.state == "out_of_frame" for o in decision.observations
    )
    if absent and not any(
        (t.crop.left, t.crop.top, t.crop.right, t.crop.bottom)
        == (0, 0, manifest.width, manifest.height)
        and _large_enough(t, policy.minimum_absence_image_scale)
        for t in tiles
    ):
        raise ValueError("absence requires a sufficiently detailed full-frame image")
    for item in decision.observations:
        if item.state == "visible" and item.center is not None:
            x, y = item.center
            if not any(
                t.crop.left <= x <= t.crop.right - 1
                and t.crop.top <= y <= t.crop.bottom - 1
                for t in tiles
            ):
                raise ValueError("confirmed ball lies outside the cited image crops")


def check_actor(
    layout: AnnotationLayout,
    actor: ArtifactRef,
    context: ReviewContext,
    manifest: ClipManifest,
) -> None:
    launch = read_document(layout, actor, Launch)
    if (
        launch.plan != context.plan
        or launch.request.adapter != "ball_review"
        or launch.request.stage != "review"
        or launch.request.subject_id != manifest.clip_id
        or launch.work_id != launch.request.work_id
        or read_document(layout, launch.request.payload, ReviewContext) != context
    ):
        raise ValueError("review actor is not the worker assigned to this ball context")


def prove_review(
    layout: AnnotationLayout,
    context: ReviewContext,
    manifest: ClipManifest,
    revision: ArtifactRef,
    labels: BallLabels,
    proof_ref: ArtifactRef,
    saved: Catalog,
) -> ReviewEvidence:
    proof = read_document(layout, proof_ref, BallReviewProof)
    if len(set(proof.telemetry)) != len(proof.telemetry):
        raise ValueError("duplicate worker telemetry proof")
    by_actor: dict[ArtifactRef, set[ArtifactRef]] = {}
    for reference in proof.telemetry:
        report = verify_report(layout, reference)
        check_actor(layout, report.launch, context, manifest)
        if report.launch in by_actor:
            raise ValueError("one worker launch has multiple telemetry reports")
        by_actor[report.launch], _ = review_images(layout, reference)
    all_shown = set().union(*by_actor.values())
    for frame in labels.frames:
        if frame.decision.reviewed:
            shown = (
                by_actor.get(frame.review_actor, set()) if frame.review_actor else set()
            )
            if any(
                saved.images[key].image not in shown for key in frame.decision.evidence
            ):
                raise ValueError(
                    "review claim images were not actually shown to its worker"
                )
    presented = ranges(
        tile.frame_index
        for key, image in saved.images.items()
        if image.image in all_shown
        for tile in saved.receipts[key].tiles
    )
    result: ReviewEvidence = ReviewEvidence(
        clip_id=manifest.clip_id,
        prediction=context.prediction,
        revision=revision,
        presented_ranges=presented,
        reviewed_ranges=ranges(
            f.frame.frame_index for f in labels.frames if f.decision.reviewed
        ),
        unresolved_ranges=ranges(
            f.frame.frame_index for f in labels.frames if not resolved(f)
        ),
        evidence=(proof_ref,),
    )
    return result
