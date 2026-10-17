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
from src.tasks.base.annotation.editing import EditCommand, ReviewContext, Revision
from src.tasks.base.annotation.images import read_rendered
from src.tasks.base.annotation.images.contracts import (
    Box,
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

from ..contracts import Box as PlayerBox
from ..contracts import PlayerPredictions
from .contracts import (
    WORKER_ADAPTER,
    WORKER_VERSION,
    FrameDecision,
    PlayerEdits,
    PlayerFrame,
    PlayerLabels,
    PlayerReviewPolicy,
    PlayerReviewProof,
    ReplaceFrame,
)
from .validation import ranges, resolved


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


ObservationBoxes = dict[tuple[int, str], PlayerBox]


def remember_observation_boxes(
    boxes: ObservationBoxes, index: int, decision: FrameDecision
) -> None:
    for item in decision.observations:
        key = (index, item.observation_id)
        if item.box is not None:
            boxes[key] = item.box
        elif item.classification in {"player", "unresolved"}:
            # Explicitly losing the location invalidates an older bbox.
            boxes.pop(key, None)


def observation_box_history(
    layout: AnnotationLayout,
    revisions: set[ArtifactRef],
    predictions: PlayerPredictions,
) -> ObservationBoxes:
    """Retain the most recent located bbox for every stable observation.

    Original proposals seed the history; later decisions replace or invalidate
    their locations exactly as they do for manually added observations.
    Include ordered edits within each revision: adding and then removing a box
    in one atomic command does not leave that bbox in the final payload.
    """
    boxes: ObservationBoxes = {
        (row.frame_index, item.observation_id): item.box
        for row in predictions.rows
        for item in row.observations
    }
    nodes = sorted(
        (read_document(layout, reference, Revision) for reference in revisions),
        key=lambda node: node.sequence,
    )
    for node in nodes:
        if node.command is not None:
            command = read_document(layout, node.command, EditCommand)
            if (
                command.expected_revision != node.previous
                or command.context != node.context
            ):
                raise ValueError("bbox history command differs from its revision")
            edits = read_document(layout, command.changes, PlayerEdits)
            for operation in edits.operations:
                if isinstance(operation, ReplaceFrame):
                    remember_observation_boxes(
                        boxes, operation.frame_index, operation.decision
                    )
        labels = read_document(layout, node.payload, PlayerLabels)
        for frame in labels.frames:
            remember_observation_boxes(boxes, frame.frame.frame_index, frame.decision)
    return boxes


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
    predictions: PlayerPredictions,
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
        labels: PlayerLabels | None = None
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
            labels = read_document(layout, node.payload, PlayerLabels)
        _drawings(request, predictions, labels)
        saved[image.sha256], receipts[image.sha256] = image, receipt
    return Catalog(saved, receipts)


def _drawings(
    request: RenderRequest, predictions: PlayerPredictions, labels: PlayerLabels | None
) -> None:
    for tile in request.tiles:
        index = tile.frame_index
        if index >= len(predictions.rows):
            raise ValueError("review image frame lies outside the native clip")
        boxes = (
            [o.box for o in predictions.rows[index].observations]
            if labels is None
            else [
                o.box
                for o in labels.frames[index].decision.observations
                if o.box is not None
            ]
        )
        expected = {(b.x1, b.y1, b.x2, b.y2) for b in boxes}
        if any(
            not isinstance(item, Box)
            or (item.left, item.top, item.right, item.bottom) not in expected
            for item in tile.drawings
        ):
            raise ValueError(
                "player overlay boxes differ from their declared annotation"
            )


def _large_enough(tile: TileReceipt, minimum: float) -> bool:
    return min(Fraction(tile.scale_x), Fraction(tile.scale_y)) >= Fraction(str(minimum))


def check_frame_images(
    frame: PlayerFrame,
    manifest: ClipManifest,
    policy: PlayerReviewPolicy,
    saved: Catalog,
    *,
    removed_boxes: tuple[PlayerBox, ...] = (),
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
    ]
    full = [
        t
        for t in tiles
        if (t.crop.left, t.crop.top, t.crop.right, t.crop.bottom)
        == (0, 0, manifest.width, manifest.height)
        and _large_enough(t, policy.minimum_full_frame_image_scale)
    ]
    detailed = [t for t in tiles if _large_enough(t, policy.minimum_image_scale)]
    if not detailed and not full:
        raise ValueError("reviewed frame lacks a sufficiently detailed native image")
    if decision.coverage == "complete" and not full:
        raise ValueError(
            "complete target inventory or empty target requires a full-frame image"
        )
    boxes = list(removed_boxes)
    for item in decision.observations:
        if item.box is not None:
            boxes.append(item.box)
    for box in boxes:
        if not any(
            t.crop.left <= box.x1
            and t.crop.top <= box.y1
            and box.x2 <= t.crop.right
            and box.y2 <= t.crop.bottom
            for t in detailed
        ):
            raise ValueError(
                "decided bbox is not fully contained in a sufficiently detailed cited crop"
            )


def check_removal_images(
    frame: PlayerFrame,
    manifest: ClipManifest,
    policy: PlayerReviewPolicy,
    saved: Catalog,
    history: ObservationBoxes,
) -> None:
    """History-dependent gate used both before edits commit and before adoption.

    Stateless payload validation cannot recover a removed observation's bbox. Its
    geometry comes from history; its images must come from this frame's current
    decision, whose review_actor is separately checked against telemetry.
    """
    boxes = []
    for item in frame.decision.observations:
        if item.classification in {
            "non_player",
            "duplicate",
        }:
            key = (frame.frame.frame_index, item.observation_id)
            if key not in history:
                raise ValueError(
                    "removed observation has no prior bbox for image verification"
                )
            boxes.append(history[key])
    if boxes:
        check_frame_images(frame, manifest, policy, saved, removed_boxes=tuple(boxes))


def check_actor(
    layout: AnnotationLayout,
    actor: ArtifactRef,
    context: ReviewContext,
    manifest: ClipManifest,
) -> None:
    launch = read_document(layout, actor, Launch)
    if (
        launch.plan != context.plan
        or launch.request.adapter != WORKER_ADAPTER
        or launch.request.adapter_version != WORKER_VERSION
        or launch.request.stage != "review"
        or launch.request.subject_id != manifest.clip_id
        or launch.work_id != launch.request.work_id
        or read_document(layout, launch.request.payload, ReviewContext) != context
    ):
        raise ValueError(
            "review actor is not the worker assigned to this player context"
        )


@dataclass(frozen=True)
class ImagePresentationProof:
    images: frozenset[ArtifactRef]
    unavailable_reason: str | None


def image_presentation_proof(
    layout: AnnotationLayout, reference: ArtifactRef
) -> ImagePresentationProof:
    """Keep authentic but unavailable past evidence explicit and recoverable.

    presented_images verifies all referenced bytes before it can report missing
    model/response provenance. Integrity errors must still propagate.
    """
    try:
        return ImagePresentationProof(
            frozenset(presented_images(layout, reference)), None
        )
    except ImagePresentationUnavailable as exc:
        return ImagePresentationProof(frozenset(), str(exc))


def prove_review(
    layout: AnnotationLayout,
    context: ReviewContext,
    manifest: ClipManifest,
    revision: ArtifactRef,
    labels: PlayerLabels,
    proof_ref: ArtifactRef,
    saved: Catalog,
) -> ReviewEvidence:
    proof = read_document(layout, proof_ref, PlayerReviewProof)
    if len(set(proof.telemetry)) != len(proof.telemetry):
        raise ValueError("duplicate worker telemetry proof")
    by_actor: dict[ArtifactRef, ImagePresentationProof] = {}
    for reference in proof.telemetry:
        report = verify_report(layout, reference)
        check_actor(layout, report.launch, context, manifest)
        if report.launch in by_actor:
            raise ValueError("one worker launch has multiple telemetry reports")
        by_actor[report.launch] = image_presentation_proof(layout, reference)
    all_shown = set().union(*(item.images for item in by_actor.values()))
    for frame in labels.frames:
        if frame.decision.reviewed:
            presentation = (
                by_actor.get(frame.review_actor) if frame.review_actor else None
            )
            shown = presentation.images if presentation is not None else frozenset()
            if any(
                saved.images[key].image not in shown for key in frame.decision.evidence
            ):
                reason = (
                    f": {presentation.unavailable_reason}"
                    if presentation is not None
                    and presentation.unavailable_reason is not None
                    else ""
                )
                raise ValueError(
                    "review claim images were not actually shown to its worker" + reason
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
