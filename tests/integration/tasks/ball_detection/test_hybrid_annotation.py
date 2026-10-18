from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.ball_detection.annotation.contracts import (
    HybridCheckpointSpec,
    HybridPredictions,
    read_predictions,
)
from src.tasks.ball_detection.annotation.hybrid import HybridPredictor
from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
from src.tasks.ball_detection.annotation.preview import preview_drawings
from src.tasks.ball_detection.annotation.review.contracts import (
    BallEdits,
    BallTeacherMask,
    ConfirmAbsence,
)
from src.tasks.ball_detection.annotation.review.validation import teacher_rows
from src.tasks.ball_detection.annotation.review.worker import BallRowPage
from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import AdoptionReceipt, FrameRange
from src.tasks.base.annotation.handoff import read_handoff
from src.tasks.base.annotation.images import read_rendered
from src.tasks.base.annotation.images.contracts import Crop, Point
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionReceipt
from src.tasks.base.annotation.workers.contracts import Launch
from src.tasks.base.annotation.workers.review.contracts import (
    HandoffRequest,
    RowsRequest,
)
from tests.support.tasks.ball_detection.annotation import Environment
from tests.support.tasks.ball_detection.hybrid_annotation import (
    SpyHybrid,
    checkpoint,
    inference,
)
from tests.support.tasks.ball_detection.review_worker import ReviewSession


@pytest.mark.parametrize("jpeg", [False, True])
@pytest.mark.parametrize("present", [False, True])
def test_tiny_hybrid_checkpoint_to_native_proposals(
    tmp_path: Path, jpeg: bool, present: bool
) -> None:
    root = tmp_path / "a"
    model = checkpoint(
        AnnotationLayout.testing(root), presence_logit=2.0 if present else -2.0
    )
    env = Environment(
        root,
        model=model,
        inference_config=inference(jpeg=jpeg),
        prediction_adapter=BallPredictionAdapter(),
        jpeg=jpeg,
        frame_count=33,
    )
    receipt = read_document(env.layout, env.context.prediction, PredictionReceipt)
    doc = read_predictions(env.layout, receipt.prediction.artifact)
    assert isinstance(doc, HybridPredictions) and len(doc.rows) == 33
    assert doc.heatmap_size == (8, 16)
    for row in doc.rows:
        assert row.presence == ("present" if present else "absent")
        assert row.heatmap_peak == pytest.approx(
            0.8807971
        )  # Ungated, also when absent.
        assert row.center == ((0.0, 0.0) if present else None)
        assert row.source_pts == env.manifest.frames[row.frame_index].source_pts
    assert all(row.heatmap == "ignore" for row in teacher_rows(env.labels()))
    assert preview_drawings(env.layout, receipt, (1,)) == (
        ((Point(x=0.0, y=0.0),) if present else ()),
    )


@pytest.mark.parametrize(
    "mismatch", ["architecture", "input_contract", "manifest", "threshold", "weights"]
)
def test_hybrid_restore_rejects_unpinned_contracts(
    tmp_path: Path, mismatch: str
) -> None:
    root = tmp_path / "a"
    model = checkpoint(AnnotationLayout.testing(root))
    env = Environment(
        root,
        model=model,
        inference_config=inference(),
        prediction_adapter=BallPredictionAdapter(),
    )
    context = read_document(
        env.layout, env.context.prediction, PredictionReceipt
    ).context
    spec = read_document(env.layout, model.configuration, HybridCheckpointSpec)
    config = inference()
    if mismatch == "threshold":
        config = config.model_copy(update={"presence_threshold": 0.7})
    elif mismatch == "weights":
        env.layout.artifact_path(model.inference_weights).write_bytes(
            b"changed weights"
        )
    else:
        update = {
            "architecture": {"architecture": {**spec.architecture, "dim": 32}},
            "input_contract": {
                "input_contract": {**spec.input_contract, "channels": "wrong"}
            },
            "manifest": {"training_manifest_digest": "0" * 64},
        }[mismatch]
        bad = write_document(
            env.layout,
            env.layout.path("campaigns", "bad-spec.json"),
            spec.model_copy(update=update),
        )
        context = context.model_copy(
            update={"model": model.model_copy(update={"configuration": bad})}
        )
    with pytest.raises(ValueError):
        HybridPredictor(env.layout, context, config)


@pytest.mark.parametrize("shown", [False, True])
def test_hybrid_absence_review_images_correction_and_proven_teacher_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shown: bool
) -> None:
    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.HybridPredictor", SpyHybrid
    )
    session = ReviewSession(
        tmp_path / "a",
        inference_config=inference(),
        prediction_adapter=BallPredictionAdapter(),
    )
    env = session.env
    proposal_receipt = read_document(
        env.layout, env.context.prediction, PredictionReceipt
    )
    original = env.layout.artifact_path(
        proposal_receipt.prediction.artifact
    ).read_bytes()
    launch = read_document(env.layout, env.actor, Launch)
    prompt = env.layout.artifact_path(launch.prepared.prompt).read_text()
    assert "Predicted absence" in prompt and "full frame" in prompt
    handoff = read_handoff(
        env.layout,
        session.call(
            "review_handoff", HandoffRequest(scope=FrameRange(start=1, stop=4))
        ),
    )
    assert "prediction=absent_unreviewed" in handoff.content.rows[0].detail
    page = read_document(
        env.layout,
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.head(), scope=FrameRange(start=1, stop=3)
            ),
        ),
        BallRowPage,
    )
    assert [row.presence for row in page.predictions] == ["absent", "present"]
    assert all(frame.decision.presence == "unknown" for frame in page.rows)
    assert all(row.heatmap == "ignore" for row in teacher_rows(env.labels()))
    image = session.image(1, 2, 3, drawing="prediction")
    request, rendered = read_rendered(env.layout, image)
    assert request.tiles[0].drawings == request.tiles[2].drawings == ()
    assert len(request.tiles[1].drawings) == 1
    assert rendered.tiles[0].source_pts == env.manifest.frames[1].source_pts
    assert env.labels().frames[1].decision.reviewed is False
    # A mixed batch cannot silently convert absent rows into visible positives.
    before = env.head()
    with pytest.raises(ValueError, match="predicted absence"):
        session.confirm(image, 1, 3)
    assert env.head() == before
    crop = env.page((1,), crop=Crop(left=0, top=0, right=32, bottom=32))
    with pytest.raises(ValueError, match="full-frame"):
        env.apply(
            BallEdits(
                operations=(
                    ConfirmAbsence(
                        scope=FrameRange(start=1, stop=2),
                        out_of_frame_tracks=(),
                        evidence=(crop.sha256,),
                    ),
                ),
                images=(crop,),
            )
        )
    # A false negative can be corrected from pixels; absence remains only a proposal.
    from src.tasks.ball_detection.annotation.review.contracts import BallObservation

    env.apply(
        BallEdits(
            operations=(
                ConfirmAbsence(
                    scope=FrameRange(start=1, stop=2),
                    out_of_frame_tracks=(),
                    evidence=(image.sha256,),
                ),
                env.replace(
                    3,
                    image,
                    observations=(
                        BallObservation(
                            track_id="ball",
                            state="visible",
                            center=(24.0, 12.0),
                            origin="manual",
                        ),
                    ),
                ),
            ),
            images=(image,),
        )
    )
    session.confirm(image, 2, 3)
    mask = teacher_rows(env.labels())
    assert [mask[i].heatmap for i in (1, 2, 3, 5)] == [
        "negative",
        "positive",
        "positive",
        "ignore",
    ]
    assert mask[1].coordinate_track is None and mask[5].coordinate_track is None
    report = session.evidence(shown=(image,) if shown else ())
    if not shown:
        with pytest.raises(ValueError, match="not actually shown"):
            session.finish(report)
        return
    outcome = session.finish(report)
    assert outcome.status == "complete" and outcome.adoption is not None
    receipt = read_document(env.layout, outcome.adoption, AdoptionReceipt)
    assert read_document(env.layout, receipt.teacher_mask, BallTeacherMask).rows == mask
    assert (
        env.layout.artifact_path(proposal_receipt.prediction.artifact).read_bytes()
        == original
    )
