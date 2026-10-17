from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    FrameRange,
    ReviewEvidence,
)
from src.tasks.base.annotation.editing import Revision
from src.tasks.base.annotation.handoff import read_handoff
from src.tasks.base.annotation.images import read_rendered
from src.tasks.base.annotation.images.contracts import Box
from src.tasks.base.annotation.storage import StorageCatalog
from src.tasks.base.annotation.telemetry.capture import (
    ImagePresentationUnavailable,
    presented_images,
    verify_report,
)
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import Launch
from src.tasks.base.annotation.workers.review.contracts import (
    EditRequest,
    HandoffRequest,
    ReviewWorkerInput,
    RowsRequest,
)
from src.tasks.player_detection.annotation.review.contracts import (
    PlayerEdits,
    PlayerReviewProof,
    PlayerTeacherMask,
)
from src.tasks.player_detection.annotation.review.worker import PlayerRowPage
from tests.support.tasks.player_detection.review_worker import ReviewSession


@pytest.fixture
def session(tmp_path: Path) -> ReviewSession:
    return ReviewSession(tmp_path / "annotation")


def test_registered_worker_uses_parent_boxes_and_bounded_rows(
    session: ReviewSession,
) -> None:
    env = session.env
    assert session.request.adapter == "player_bbox_review"
    protected = {
        r
        for p in StorageCatalog(env.layout).snapshot().protections
        for r in p.references
    }
    assert env.manifest.media in protected and env.context.prediction in protected
    image = session.image(1, drawing="prediction")
    request, receipt = read_rendered(env.layout, image)
    assert request.tiles[0].drawings == (
        Box(left=4.0, top=3.0, right=24.0, bottom=30.0),
        Box(left=38.0, top=5.0, right=60.0, bottom=32.0),
    )
    assert receipt.tiles[0].source_pts == env.manifest.frames[1].source_pts
    session.confirm(image, 1, 2)
    page = read_document(
        env.layout,
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.head(), scope=FrameRange(start=1, stop=2)
            ),
        ),
        PlayerRowPage,
    )
    assert page.rows == env.labels().frames[1:2]
    assert page.proposals[0].frame_index == page.rows[0].frame.frame_index
    assert {p.observation_id for p in page.proposals[0].observations} == {
        o.proposal_id for o in page.rows[0].decision.observations
    }
    assert page.proposals[0].observations[0].confidence == 0.9
    assert page.image_proof == ("pending_current_attempt",)
    with pytest.raises(ValueError, match="eight"):
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.head(), scope=FrameRange(start=1, stop=10)
            ),
        )
    with pytest.raises(ValueError, match="stale"):
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.initial, scope=FrameRange(start=1, stop=2)
            ),
        )


def test_worker_completion_validates_images_and_seals_mask(
    session: ReviewSession,
) -> None:
    env = session.env
    session.confirm(session.image(1), 1, 2)
    outcome = session.finish(session.evidence())
    assert outcome.status == "complete" and outcome.adoption is not None
    adoption = read_document(env.layout, outcome.adoption, AdoptionReceipt)
    mask = read_document(env.layout, adoption.teacher_mask, PlayerTeacherMask)
    assert (
        mask.rows[1].supervision == "positive" and mask.rows[2].supervision == "ignore"
    )
    assert env.store.current(env.context_ref, session.services)[1] == outcome.adoption


def test_retry_retains_reviewed_boxes_and_images_from_cancelled_attempt(
    session: ReviewSession,
) -> None:
    env = session.env
    first = session.image(1)
    edit = session.confirm(first, 1, 2)
    session.evidence(status="cancelled")
    session.state("cancelled")
    session.open(2)
    launch = read_document(env.layout, env.actor, Launch)
    node = read_document(env.layout, edit.revision, Revision)
    assert (
        edit.revision in launch.prepared.inputs
        and node.payload in launch.prepared.inputs
    )
    assert (
        first.image in launch.prepared.inputs and edit.command in launch.prepared.inputs
    )
    handoff = read_handoff(
        env.layout,
        session.call(
            "review_handoff", HandoffRequest(scope=FrameRange(start=1, stop=2))
        ),
    )
    assert "image_proof=verified" in handoff.content.rows[0].detail
    session.confirm(session.image(3), 3, 4)
    assert session.finish(session.evidence()).status == "complete"


def test_missing_prior_proof_is_visible_and_cannot_be_reassigned(
    session: ReviewSession,
) -> None:
    session.confirm(session.image(1), 1, 2)
    session.evidence(shown=(), status="cancelled")
    session.state("cancelled")
    session.open(2)
    handoff = read_handoff(
        session.env.layout,
        session.call(
            "review_handoff", HandoffRequest(scope=FrameRange(start=1, stop=2))
        ),
    )
    assert "image_proof=missing" in handoff.content.rows[0].detail
    # Seeing the same image in the second worker does not rewrite the first author.
    with pytest.raises(ValueError, match="actually shown"):
        session.finish(session.evidence(shown=session.env.labels().images))


def test_worker_can_defer_without_adopting_unreviewed_frames(
    session: ReviewSession,
) -> None:
    outcome = session.finish(session.evidence(shown=()), status="deferred")
    assert outcome.status == "deferred" and outcome.adoption is None
    assert (
        session.env.store.current(session.env.context_ref, session.services)[1] is None
    )


def reinspect(session: ReviewSession) -> RenderedImage:
    env = session.env
    image = session.image(1, drawing="revision")
    edits = PlayerEdits(
        operations=(
            env.replace(
                1,
                image,
                observations=env.labels().frames[1].decision.observations,
            ),
        ),
        images=(image,),
    )
    changes = write_document(
        env.layout, session.directory / "work/reinspect.json", edits
    )
    session.call(
        "review_edit",
        EditRequest(
            expected_revision=env.head(),
            changes=changes,
            reason="reinspect missing prior image proof",
        ),
    )
    return image


@pytest.mark.parametrize(
    "fault, reason",
    [
        ("missing_model", "logged model is missing"),
        ("conflicting_response_id", "conflicting response IDs"),
    ],
)
@pytest.mark.parametrize("reinspection", ["none", "unseen", "seen"])
def test_unavailable_prior_proof_recovers_only_after_reinspection(
    session: ReviewSession,
    fault: Literal["missing_model", "conflicting_response_id"],
    reason: str,
    reinspection: str,
) -> None:
    env = session.env
    image = session.image(1)
    session.confirm(image, 1, 2)
    prior = session.evidence(status="cancelled", telemetry_fault=fault)
    with pytest.raises(ImagePresentationUnavailable, match=reason):
        presented_images(env.layout, prior)
    session.state("cancelled")
    session.open(2)
    request = read_document(env.layout, session.input, ReviewWorkerInput)
    handoff = read_handoff(env.layout, request.initial_handoff)
    assert "image_proof=missing" in handoff.content.rows[0].detail
    assert reason in handoff.content.rows[0].detail
    page = read_document(
        env.layout,
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.head(),
                scope=FrameRange(start=1, stop=2),
            ),
        ),
        PlayerRowPage,
    )
    assert page.image_proof == ("missing",)
    missing_reason = page.image_proof_reasons[0]
    assert missing_reason is not None and reason in missing_reason
    if reinspection != "none":
        image = reinspect(session)
    report = session.evidence(shown=() if reinspection == "unseen" else (image,))
    if reinspection != "seen":
        with pytest.raises(ValueError, match="actually shown"):
            session.finish(report)
        assert env.store.current(env.context_ref, session.services)[1] is None
    else:
        outcome = session.finish(report)
        assert outcome.status == "complete" and outcome.adoption is not None
        adoption = read_document(env.layout, outcome.adoption, AdoptionReceipt)
        evidence = read_document(env.layout, adoption.review_evidence, ReviewEvidence)
        proof = read_document(env.layout, evidence.evidence[0], PlayerReviewProof)
        assert set(proof.telemetry) == {prior, report}
        assert env.labels().frames[1].review_actor == env.actor


@pytest.mark.parametrize("boundary", ["prepare", "adoption"])
@pytest.mark.parametrize("artifact", ["report", "page"])
def test_unavailable_prior_proof_still_rejects_hash_corruption(
    session: ReviewSession,
    boundary: str,
    artifact: str,
) -> None:
    env = session.env
    session.confirm(session.image(1), 1, 2)
    prior = session.evidence(status="cancelled", telemetry_fault="missing_model")
    report = verify_report(env.layout, prior)
    session.state("cancelled")
    current = None
    if boundary == "adoption":
        session.open(2)
        current = session.evidence(shown=(reinspect(session),))
    ref = prior if artifact == "report" else report.pages[0]
    path = env.layout.artifact_path(ref)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash changed"):
        if boundary == "prepare":
            session.open(2)
        else:
            assert current is not None
            session.finish(current)


@pytest.mark.parametrize("fault", ["missing_model", "conflicting_response_id"])
def test_current_attempt_still_requires_verifiable_image_provenance(
    session: ReviewSession,
    fault: Literal["missing_model", "conflicting_response_id"],
) -> None:
    session.confirm(session.image(1), 1, 2)
    session.evidence(status="cancelled")
    session.state("cancelled")
    session.open(2)
    report = session.evidence(telemetry_fault=fault)
    with pytest.raises(ImagePresentationUnavailable):
        session.finish(report)


def test_worker_scratch_edits_are_frozen_before_adoption(
    session: ReviewSession,
) -> None:
    session.confirm(session.image(1), 1, 2)
    for path in (session.directory / "work").glob("*.json"):
        path.write_text("modified scratch")
    assert session.finish(session.evidence()).status == "complete"
