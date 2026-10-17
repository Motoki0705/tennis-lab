"""Ball's implementation of common revision, handoff and adoption interfaces."""

from __future__ import annotations

from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
    Document,
    FrameRange,
    ReviewEvidence,
)
from src.tasks.base.annotation.editing import (
    ChangeResult,
    EditCommand,
    ReviewContext,
    Revision,
)
from src.tasks.base.annotation.handoff import Boundary, HandoffContent, SummaryRow
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import (
    BallEdits,
    BallFrame,
    BallLabels,
    BallReviewProof,
    BallTeacherMask,
    FrameDecision,
)
from .edits import apply_operations, brief
from .evidence import catalog, check_actor, check_frame_images, lineage, prove_review
from .validation import (
    check_acceptance,
    inputs,
    ranges,
    resolved,
    teacher_rows,
    validate_labels,
)


class BallReviewAdapter:
    task = "ball"

    def initialize(
        self, layout: AnnotationLayout, context: ReviewContext, manifest: ClipManifest
    ) -> BallLabels:
        policy_ref, _, _ = inputs(layout, context, manifest)
        return BallLabels(
            clip_id=manifest.clip_id,
            prediction=context.prediction,
            policy=policy_ref,
            images=(),
            frames=tuple(
                BallFrame(
                    frame=frame,
                    target=manifest.target_range.start
                    <= frame.frame_index
                    < manifest.target_range.stop,
                    decision=FrameDecision(
                        reviewed=False,
                        presence="unknown",
                        observations=(),
                        evidence=(),
                        notes=(),
                        boundaries=(),
                    ),
                    review_actor=None,
                )
                for frame in manifest.frames
            ),
        )

    def validate(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        payload: Document,
    ) -> None:
        labels = BallLabels.model_validate_json(payload.model_dump_json())
        policy_ref, policy, predictions = inputs(layout, context, manifest)
        validate_labels(labels, context, manifest, policy_ref, policy, predictions)
        saved = catalog(layout, context, labels.images, predictions)
        for actor in {
            f.review_actor for f in labels.frames if f.review_actor is not None
        }:
            check_actor(layout, actor, context, manifest)
        for frame in labels.frames:
            check_frame_images(frame, manifest, policy, saved)

    def apply(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        current_payload: ArtifactRef,
        command: EditCommand,
    ) -> ChangeResult:
        labels = read_document(layout, current_payload, BallLabels)
        self.validate(layout, context, manifest, labels)
        previous = read_document(layout, command.expected_revision, Revision)
        if (
            previous.payload != current_payload
            or read_document(layout, command.context, ReviewContext) != context
            or previous.context != command.context
        ):
            raise ValueError("edit does not address the current ball payload/context")
        check_actor(layout, command.actor, context, manifest)
        edits = read_document(layout, command.changes, BallEdits)
        if len({image.sha256 for image in edits.images}) != len(edits.images):
            raise ValueError("edit contains duplicate review images")
        merged = {image.sha256: image for image in (*labels.images, *edits.images)}
        _, policy, predictions = inputs(layout, context, manifest)
        saved = catalog(
            layout,
            context,
            tuple(merged.values()),
            predictions,
            allowed_revisions=lineage(layout, context, command.expected_revision),
        )
        result = apply_operations(
            labels, edits, manifest, policy, predictions, saved, command.actor
        )
        self.validate(layout, context, manifest, result.payload)
        return result

    def _adoption_parts(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        revision: ArtifactRef,
        proof: ArtifactRef,
    ) -> tuple[BallTeacherMask, ReviewEvidence]:
        node = read_document(layout, revision, Revision)
        if read_document(layout, node.context, ReviewContext) != context:
            raise ValueError("adoption revision belongs to another ball context")
        labels = read_document(layout, node.payload, BallLabels)
        self.validate(layout, context, manifest, labels)
        policy_ref, policy, predictions = inputs(layout, context, manifest)
        saved = catalog(
            layout,
            context,
            labels.images,
            predictions,
            allowed_revisions=lineage(layout, context, revision),
        )
        evidence = prove_review(
            layout, context, manifest, revision, labels, proof, saved
        )
        check_acceptance(labels, policy)
        mask = BallTeacherMask(
            clip_id=manifest.clip_id,
            revision=revision,
            policy=policy_ref,
            rows=teacher_rows(labels),
        )
        return mask, evidence

    def prepare_adoption(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        revision: ArtifactRef,
        telemetry: tuple[ArtifactRef, ...],
        *,
        reason: str,
    ) -> ArtifactRef:
        """Create machine-validated receipts; the caller then seals this exact head."""
        if not reason.strip():
            raise ValueError("adoption requires a reason")
        directory = layout.path(
            "annotations", "ball", manifest.clip_id, "adoption-evidence"
        )
        proof = BallReviewProof(telemetry=telemetry)
        proof_ref = write_document(layout, directory / (proof.sha256 + ".json"), proof)
        mask, evidence = self._adoption_parts(
            layout, context, manifest, revision, proof_ref
        )
        mask_ref = write_document(layout, directory / (mask.sha256 + ".json"), mask)
        evidence_ref = write_document(
            layout, directory / (evidence.sha256 + ".json"), evidence
        )
        receipt = AdoptionReceipt(
            task="ball",
            clip_id=manifest.clip_id,
            revision=revision,
            review_evidence=evidence_ref,
            policy=mask.policy,
            teacher_mask=mask_ref,
            reason=reason,
        )
        reference: ArtifactRef = write_document(
            layout, directory / (receipt.sha256 + ".json"), receipt
        )
        return reference

    def validate_adoption(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        revision: ArtifactRef,
        receipt: AdoptionReceipt,
    ) -> None:
        if not receipt.reason.strip():
            raise ValueError("adoption requires a reason")
        evidence = read_document(layout, receipt.review_evidence, ReviewEvidence)
        if len(evidence.evidence) != 1:
            raise ValueError(
                "ball adoption requires one complete worker proof document"
            )
        mask, expected = self._adoption_parts(
            layout, context, manifest, revision, evidence.evidence[0]
        )
        if (
            (receipt.task, receipt.clip_id, receipt.revision, receipt.policy)
            != ("ball", manifest.clip_id, revision, mask.policy)
            or evidence != expected
            or read_document(layout, receipt.teacher_mask, BallTeacherMask) != mask
        ):
            raise ValueError(
                "ball adoption coverage or teacher mask differs from validated decisions"
            )

    def summarize(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        payload: ArtifactRef,
        scope: FrameRange,
        after_row: str | None,
    ) -> HandoffContent:
        labels = read_document(layout, payload, BallLabels)
        self.validate(layout, context, manifest, labels)
        if scope.stop > len(labels.frames) or scope.stop - scope.start > 64:
            raise ValueError("ball handoff must contain at most 64 native frames")
        start = scope.start
        if after_row is not None:
            if (
                not after_row.isdecimal()
                or str(int(after_row)) != after_row
                or not scope.start <= int(after_row) < scope.stop
            ):
                raise ValueError(
                    "ball handoff cursor is outside its native frame scope"
                )
            start = int(after_row) + 1
        selected = labels.frames[scope.start : scope.stop]
        return HandoffContent(
            rows=tuple(
                SummaryRow(
                    frame_index=f.frame.frame_index,
                    row_id=str(f.frame.frame_index),
                    state="unreviewed"
                    if not f.decision.reviewed
                    else "unresolved"
                    if not resolved(f)
                    else f.decision.presence,
                    detail=f"digest={f.sha256} {brief(f)}",
                )
                for f in labels.frames[start : scope.stop]
            ),
            total_rows=len(selected),
            remaining_rows=0,
            next_row_cursor=None,
            reviewed_ranges=ranges(
                f.frame.frame_index for f in selected if f.decision.reviewed
            ),
            unresolved_ranges=ranges(
                f.frame.frame_index for f in selected if not resolved(f)
            ),
            boundaries=tuple(
                Boundary(
                    frame_index=f.frame.frame_index,
                    kind="_".join(f.decision.boundaries),
                )
                for f in selected
                if f.decision.boundaries
            ),
            next_actions=(
                "Inspect unresolved frames and record explicit decisions; preserve notes and boundaries."
                if any(not resolved(f) for f in selected)
                else "Check clip-wide policy coverage and worker image proof before adoption.",
            ),
        )
