from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.tasks.ball_detection.annotation.review.contracts import (
    BallEdits,
    BallLabels,
    BallObservation,
    BallReviewPolicy,
    BallTeacherMask,
    ConfirmAbsence,
    ConfirmPredictions,
    FrameDecision,
    Interpolate,
    Interpolation,
    Note,
    ReplaceFrame,
)
from src.tasks.ball_detection.annotation.review.validation import teacher_rows
from src.tasks.base.annotation.artifacts import (
    read_document,
)
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    FrameRange,
    ReviewEvidence,
)
from src.tasks.base.annotation.editing import (
    Revision,
    RevisionConflict,
)
from src.tasks.base.annotation.editing.contracts import EditReceipt
from src.tasks.base.annotation.handoff import build_handoff, read_handoff
from src.tasks.base.annotation.images.contracts import Crop, Point
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from tests.support.tasks.ball_detection.annotation import Environment


@pytest.fixture
def env(tmp_path: Path) -> Environment:
    return Environment(tmp_path / "annotation")


def visible(x: float = 16.0, *, track: str = "ball") -> BallObservation:
    return BallObservation(
        track_id=track, state="visible", center=(x, 12.0), origin="manual"
    )


@pytest.mark.parametrize("orphan", [False, True])
def test_adoption_retains_render_requests_for_uncited_telemetry_images(
    env: Environment, orphan: bool,
) -> None:
    from src.tasks.base.annotation.images import contracts as image_contracts
    from src.tasks.base.annotation.images import read_rendered
    from src.tasks.base.annotation.plan import references
    from src.tasks.base.annotation.telemetry.contracts import TelemetryReport
    from tests.support.annotation_storage import (
        assert_expired_but_protected,
        reference_document_types,
    )

    cited = env.page((1,))
    browsed = env.page((2,), revision=env.initial)
    env.apply(BallEdits(
        operations=(env.replace(1, cited, observations=(visible(),)),), images=(cited,),
    ))
    assert browsed not in env.labels().images
    telemetry = env.telemetry(
        (cited, browsed),
        shown=(cited,) if orphan else (cited, browsed),
        orphaned=(browsed,) if orphan else (),
    )
    report = read_document(env.layout, telemetry, TelemetryReport)
    assert report.orphan_output_count == int(orphan)
    adopted = env.adapter.prepare_adoption(
        env.layout, env.context, env.manifest, env.head(), (telemetry,),
        reason="retain browsed but uncited image evidence",
    )
    env.store.seal(env.head(), adopted, env.adapter)
    request, receipt = read_rendered(env.layout, browsed)
    assert {type(request), type(receipt)} == reference_document_types(image_contracts)
    assert_expired_but_protected(
        env.layout,
        (browsed.image, browsed.metadata, *references(receipt), *references(request)),
        "adoption",
    )
    env.store.seal(env.head(), adopted, env.adapter)


def test_seal_and_reference_protection_roll_back_together(
    env: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.base.annotation.storage.catalog import StorageCatalog

    image = env.page((1,))
    env.apply(BallEdits(
        operations=(env.replace(1, image, observations=(visible(),)),), images=(image,),
    ))
    adopted = env.adapter.prepare_adoption(
        env.layout, env.context, env.manifest, env.head(), (env.telemetry((image,)),),
        reason="atomic protection fixture",
    )
    before = StorageCatalog(env.layout).snapshot().digest
    original = env.store.catalog.protect

    def crash(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("after protection insert")

    with monkeypatch.context() as patch:
        patch.setattr(env.store.catalog, "protect", crash)
        with pytest.raises(RuntimeError, match="after protection"):
            env.store.seal(env.head(), adopted, env.adapter)
    assert StorageCatalog(env.layout).snapshot().digest == before
    with env.store.registry.transaction() as db:
        assert db.execute("SELECT adoption FROM annotation_revision_heads").fetchone()[0] is None
        assert db.execute("SELECT state FROM annotation_tasks").fetchone()[0] == "reviewing"
    env.store.seal(env.head(), adopted, env.adapter)
    first = StorageCatalog(env.layout).snapshot().digest
    env.store.seal(env.head(), adopted, env.adapter)
    assert StorageCatalog(env.layout).snapshot().digest == first


def test_initial_labels_preserve_all_native_pts_without_copying_proposals(
    env: Environment,
) -> None:
    labels = env.labels()
    assert tuple(f.frame for f in labels.frames) == env.manifest.frames
    assert [f.target for f in labels.frames] == [False] + [True] * 30 + [False]
    assert all(
        not f.decision.reviewed and not f.decision.observations for f in labels.frames
    )
    assert all(
        row.heatmap == "ignore" and row.coordinate_track is None
        for row in teacher_rows(labels)
    )
    assert (
        labels.prediction == env.context.prediction
        and env.store.initialize(env.context, env.adapter) == env.initial
    )


def test_bulk_confirmation_keeps_proposal_identity_history_and_idempotent_receipt(
    env: Environment,
) -> None:
    original = env.layout.artifact_path(env.context.prediction).read_bytes()
    image = env.page((1, 2, 3))
    command = env.command(
        BallEdits(
            operations=(
                ConfirmPredictions(
                    scope=FrameRange(start=1, stop=4),
                    track_id="ball",
                    evidence=(image.sha256,),
                ),
            ),
            images=(image,),
        )
    )
    receipt_ref = env.store.apply(command, env.adapter)
    assert env.store.apply(command, env.adapter) == receipt_ref
    receipt = read_document(env.layout, receipt_ref, EditReceipt)
    assert receipt.summary.changed_rows == 3 and receipt.summary.unresolved_rows == 27
    assert env.layout.artifact_path(env.context.prediction).read_bytes() == original
    assert not env.labels(env.initial).frames[1].decision.reviewed
    labels = env.labels()
    assert all(
        f.decision.observations[0].origin == "model_confirmed"
        and f.review_actor == env.actor
        for f in labels.frames[1:4]
    )
    assert [row.coordinate_track for row in teacher_rows(labels)[1:5]] == [
        "ball",
        "ball",
        "ball",
        None,
    ]


def test_failed_bulk_is_atomic_and_cannot_copy_first_frame_review_to_others(
    env: Environment,
) -> None:
    image = env.page((1, 2))
    with pytest.raises(ValueError, match="image references"):
        env.apply(
            BallEdits(
                operations=(
                    ConfirmPredictions(
                        scope=FrameRange(start=1, stop=4),
                        track_id="ball",
                        evidence=(image.sha256,),
                    ),
                ),
                images=(image,),
            )
        )
    assert env.head() == env.initial


def test_bulk_cannot_overwrite_reviewed_rows_and_explicit_replacement_preserves_notes(
    env: Environment,
) -> None:
    image = env.page((1, 2))
    note = Note(note_id="blur", text="location unclear", blocking=True)
    env.apply(
        BallEdits(
            operations=(
                env.replace(1, image, observations=(visible(),), notes=(note,)),
            ),
            images=(image,),
        )
    )
    noted = env.head()
    with pytest.raises(ValueError, match="cannot overwrite"):
        env.apply(
            BallEdits(
                operations=(
                    ConfirmAbsence(
                        scope=FrameRange(start=1, stop=3),
                        out_of_frame_tracks=(),
                        evidence=(image.sha256,),
                    ),
                )
            )
        )
    assert env.head() == noted
    stale = env.replace(1, image, presence="absent")
    env.apply(
        BallEdits(operations=(env.replace(1, image, observations=(visible(18.0),)),))
    )
    with pytest.raises(ValueError, match="acknowledge"):
        env.apply(BallEdits(operations=(stale,)))
    assert env.labels(noted).frames[1].decision.notes == (note,)
    assert not env.labels().frames[1].decision.notes


def test_state_masks_keep_absence_estimates_uncertainty_context_and_multiple_balls_distinct(
    env: Environment,
) -> None:
    image = env.page(tuple(range(12)))
    observations = {
        0: (visible(),),
        1: (visible(),),
        2: (),
        3: (
            BallObservation(
                track_id="ball", state="out_of_frame", center=None, origin="manual"
            ),
        ),
        4: (),
        5: (
            BallObservation(
                track_id="ball", state="unresolved", center=None, origin="manual"
            ),
        ),
        6: (
            BallObservation(
                track_id="ball", state="occluded", center=(16.0, 12.0), origin="manual"
            ),
        ),
        7: (
            BallObservation(
                track_id="ball", state="occluded", center=None, origin="manual"
            ),
        ),
        8: (visible(track="a"), visible(20.0, track="b")),
        9: (
            visible(track="a"),
            BallObservation(
                track_id="b", state="unresolved", center=None, origin="manual"
            ),
        ),
        10: (visible(),),
    }
    ops = tuple(
        env.replace(
            i,
            image,
            observations=items,
            presence="absent" if i in {2, 3} else "unknown" if i == 4 else "present",
            notes=(Note(note_id="check", text="needs resolution", blocking=True),)
            if i == 10
            else (),
        )
        for i, items in observations.items()
    )
    env.apply(BallEdits(operations=ops, images=(image,)))
    rows = teacher_rows(env.labels())
    assert rows[0].heatmap == "ignore" and rows[0].reviewed
    assert rows[1].heatmap == "positive" and rows[1].coordinate_track == "ball"
    assert all(
        rows[i].heatmap == "negative" and rows[i].coordinate_track is None
        for i in (2, 3)
    )
    assert all(
        rows[i].heatmap == "ignore" and rows[i].coordinate_track is None
        for i in (4, 5, 6, 7, 9, 10, 11)
    )
    assert (
        rows[8].heatmap == "positive"
        and rows[8].positive_tracks == ("a", "b")
        and rows[8].coordinate_track is None
    )
    handoff = read_handoff(
        env.layout,
        build_handoff(
            env.store, env.context_ref, env.adapter, FrameRange(start=4, stop=8)
        ),
    )
    assert handoff.content.reviewed_ranges == (FrameRange(start=4, stop=8),)
    assert handoff.content.unresolved_ranges == (
        FrameRange(start=4, stop=6),
        FrameRange(start=7, stop=8),
    )


@pytest.mark.parametrize(
    "problem",
    [
        "nan",
        "duplicate_track",
        "outside",
        "fake_model",
        "unknown_with_point",
        "missing_time",
        "target",
    ],
)
def test_invalid_label_semantics_and_identity_are_rejected(
    env: Environment, problem: str
) -> None:
    image = env.page((1,))
    if problem in {"missing_time", "target"}:
        data = env.labels().model_dump(mode="json")
        if problem == "missing_time":
            data["frames"][1]["frame"]["source_pts"] += 1
        else:
            data["frames"][1]["target"] = False
        bad = BallLabels.model_validate_json(json.dumps(data))
        with pytest.raises(ValueError, match="native frame/PTS"):
            env.adapter.validate(env.layout, env.context, env.manifest, bad)
    else:
        with pytest.raises(ValueError):
            items: tuple[BallObservation, ...] = (
                visible(
                    float("nan")
                    if problem == "nan"
                    else 100.0
                    if problem == "outside"
                    else 18.0
                ),
            )
            if problem == "duplicate_track":
                items += items
            if problem == "fake_model":
                items = (items[0].model_copy(update={"origin": "model_confirmed"}),)
            env.apply(
                BallEdits(
                    operations=(
                        env.replace(
                            1,
                            image,
                            observations=items,
                            presence="unknown"
                            if problem == "unknown_with_point"
                            else "present",
                        ),
                    ),
                    images=(image,),
                )
            )
    assert env.head() == env.initial


def test_crop_evidence_cannot_claim_absence_or_a_point_outside_its_pixels(
    env: Environment,
) -> None:
    image = env.page((1,), crop=Crop(left=0, top=0, right=10, bottom=20))
    for operation, error in [
        (
            ConfirmAbsence(
                scope=FrameRange(start=1, stop=2),
                out_of_frame_tracks=(),
                evidence=(image.sha256,),
            ),
            "full-frame",
        ),
        (env.replace(1, image, observations=(visible(),)), "outside"),
    ]:
        with pytest.raises(ValueError, match=error):
            env.apply(BallEdits(operations=(operation,), images=(image,)))
    assert env.head() == env.initial


def test_overlay_must_match_its_declared_prediction(env: Environment) -> None:
    image = env.page((1,), markers=(Point(x=20.0, y=12.0),))
    with pytest.raises(ValueError, match="overlay markers"):
        env.apply(
            BallEdits(
                operations=(env.replace(1, image, observations=(visible(),)),),
                images=(image,),
            )
        )


def reviewed_interval(env: Environment) -> RenderedImage:
    image = env.page((2, 3, 4, 5))
    operations = tuple(
        env.replace(
            i,
            image,
            observations=(visible(8.0 if i == 2 else 24.0),)
            if i in {2, 5}
            else (
                BallObservation(
                    track_id="ball", state="unresolved", center=None, origin="manual"
                ),
            ),
        )
        for i in range(2, 6)
    )
    env.apply(BallEdits(operations=operations, images=(image,)))
    return image


def interpolate() -> Interpolate:
    return Interpolate(track_id="ball", endpoints=Interpolation(start=2, end=5))


def test_interpolation_uses_real_pts_and_endpoint_changes_require_atomic_recomputation(
    env: Environment,
) -> None:
    image = reviewed_interval(env)
    env.apply(BallEdits(operations=(interpolate(),)))
    labels = env.labels()
    first, last = labels.frames[2].frame.source_pts, labels.frames[5].frame.source_pts
    for i in (3, 4):
        item = labels.frames[i].decision.observations[0]
        assert item.center == pytest.approx(
            (
                8 + (labels.frames[i].frame.source_pts - first) / (last - first) * 16,
                12.0,
            )
        )
        assert (
            item.origin == "interpolation"
            and teacher_rows(labels)[i].heatmap == "ignore"
        )
    head = env.head()
    change = env.replace(2, image, observations=(visible(4.0),))
    with pytest.raises(ValueError, match="stale endpoint"):
        env.apply(BallEdits(operations=(change,)))
    assert env.head() == head
    env.apply(BallEdits(operations=(change, interpolate())))
    assert (
        env.labels().frames[3].decision.observations[0].center
        != labels.frames[3].decision.observations[0].center
    )


@pytest.mark.parametrize(
    "boundary,index",
    [("cut", 2), ("hit", 3), ("bounce", 4), ("track_change", 5), ("unknown_break", 2)],
)
def test_interpolation_never_crosses_event_or_track_boundaries_including_endpoints(
    env: Environment, boundary: str, index: int
) -> None:
    image = reviewed_interval(env)
    frame = env.labels().frames[index]
    env.apply(
        BallEdits(
            operations=(
                env.replace(
                    index,
                    image,
                    observations=frame.decision.observations,
                    boundaries=(boundary,),
                ),
            )
        )
    )
    head = env.head()
    with pytest.raises(ValueError, match="boundary"):
        env.apply(BallEdits(operations=(interpolate(),)))
    assert env.head() == head


@pytest.mark.parametrize("state", ["visible", "absent", "unreviewed"])
def test_interpolation_preserves_confirmed_interior_and_requires_actual_review(
    env: Environment, state: str
) -> None:
    image = reviewed_interval(env)
    if state == "unreviewed":
        operation = ReplaceFrame(
            frame_index=3,
            expected_frame_digest=env.labels().frames[3].sha256,
            decision=FrameDecision(
                reviewed=False,
                presence="unknown",
                observations=(),
                evidence=(),
                notes=(),
                boundaries=(),
            ),
        )
    else:
        operation = env.replace(
            3,
            image,
            observations=(visible(),) if state == "visible" else (),
            presence="present" if state == "visible" else "absent",
        )
    env.apply(BallEdits(operations=(operation,)))
    head = env.head()
    with pytest.raises(ValueError, match="overwrite|unreviewed"):
        env.apply(BallEdits(operations=(interpolate(),)))
    assert env.head() == head


@pytest.mark.parametrize("maximum", [None, 0.01])
def test_interpolation_is_limited_by_explicit_policy(
    tmp_path: Path, maximum: float | None
) -> None:
    env = Environment(tmp_path / "annotation", maximum_interpolation_seconds=maximum)
    reviewed_interval(env)
    with pytest.raises(ValueError, match="disabled|time gap"):
        env.apply(BallEdits(operations=(interpolate(),)))


def test_unrelated_revision_images_are_rejected_even_for_same_clip(
    env: Environment,
) -> None:
    initial = read_document(env.layout, env.initial, Revision)
    sibling = env.save(
        Revision(
            context=initial.context,
            sequence=1,
            previous=env.initial,
            payload=initial.payload,
            command=env.blob,
        )
    )
    image = env.page((1,), revision=sibling)
    with pytest.raises(ValueError, match="different or future revision"):
        env.apply(
            BallEdits(
                operations=(env.replace(1, image, observations=(visible(),)),),
                images=(image,),
            )
        )


def test_adoption_proves_worker_images_derives_masks_and_seals_exact_revision(
    env: Environment,
) -> None:
    image = env.page((0, 1, 2, 3, 4))
    env.apply(
        BallEdits(
            operations=(
                env.replace(1, image, observations=(visible(),)),
                env.replace(2, image, presence="absent"),
                env.replace(3, image, presence="unknown"),
            ),
            images=(image,),
        )
    )
    telemetry = env.telemetry((image,))
    adopted = env.adapter.prepare_adoption(
        env.layout,
        env.context,
        env.manifest,
        env.head(),
        (telemetry,),
        reason="synthetic partial policy verified",
    )
    receipt = read_document(env.layout, adopted, AdoptionReceipt)
    evidence = read_document(env.layout, receipt.review_evidence, ReviewEvidence)
    mask = read_document(env.layout, receipt.teacher_mask, BallTeacherMask)
    assert evidence.presented_ranges == (FrameRange(start=0, stop=5),)
    assert evidence.reviewed_ranges == (FrameRange(start=1, stop=4),)
    assert evidence.unresolved_ranges == (
        FrameRange(start=0, stop=1),
        FrameRange(start=3, stop=32),
    )
    assert (
        mask.rows[1].coordinate_track == "ball" and mask.rows[2].heatmap == "negative"
    )
    assert mask.rows[2].coordinate_track is None and mask.rows[3].heatmap == "ignore"
    env.store.seal(env.head(), adopted, env.adapter)
    from src.tasks.base.annotation.editing.contracts import EditCommand
    from src.tasks.base.annotation.images.contracts import RenderReceipt
    from src.tasks.base.annotation.telemetry.contracts import (
        ResponsePage,
        TelemetryReport,
    )
    from src.tasks.base.annotation.workers.contracts import ExitReceipt
    from tests.support.annotation_storage import assert_expired_but_protected

    revision = read_document(env.layout, env.head(), Revision)
    command = read_document(env.layout, revision.command, EditCommand)
    report = read_document(env.layout, telemetry, TelemetryReport)
    page = read_document(env.layout, report.pages[0], ResponsePage)
    execution = read_document(env.layout, report.execution, ExitReceipt)
    rendered = read_document(env.layout, image.metadata, RenderReceipt)
    assert_expired_but_protected(
        env.layout,
        (adopted, env.initial, env.head(), revision.payload, revision.command,
         command.changes, command.actor, env.context.manifest, env.context.prediction,
         env.manifest.media, env.manifest.source.media, receipt.teacher_mask,
         image.image, image.metadata, rendered.request, telemetry, report.pages[0],
         page.entries[0].record, page.entries[0].page, report.index_page,
         report.execution, execution.events, execution.errors),
        "adoption",
    )
    with pytest.raises(RevisionConflict, match="sealed"):
        env.apply(BallEdits(operations=(env.replace(1, image, presence="absent"),)))


@pytest.mark.parametrize("mode", ["generated_only", "other_worker"])
def test_generated_or_other_workers_images_cannot_prove_review(
    env: Environment, mode: str
) -> None:
    image = env.page((1,))
    env.apply(
        BallEdits(
            operations=(env.replace(1, image, observations=(visible(),)),),
            images=(image,),
        )
    )
    actor = env.new_actor("other") if mode == "other_worker" else env.actor
    report = env.telemetry(
        (image,), shown=() if mode == "generated_only" else (image,), actor=actor
    )
    with pytest.raises(ValueError, match="not actually shown"):
        env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (report,),
            reason="must fail",
        )
    assert env.store.current(env.context_ref, env.adapter)[1] is None


def test_partial_review_is_allowed_only_by_explicit_thresholds_and_proposal_policy(
    tmp_path: Path,
) -> None:
    env = Environment(
        tmp_path / "annotation",
        minimum_reviewed_target_fraction=1.0,
        allow_prediction_confirmation=False,
    )
    image = env.page((1,))
    with pytest.raises(ValueError, match="disabled by policy"):
        env.apply(
            BallEdits(
                operations=(
                    ConfirmPredictions(
                        scope=FrameRange(start=1, stop=2),
                        track_id="ball",
                        evidence=(image.sha256,),
                    ),
                ),
                images=(image,),
            )
        )
    env.apply(
        BallEdits(
            operations=(env.replace(1, image, observations=(visible(),)),),
            images=(image,),
        )
    )
    report = env.telemetry((image,))
    with pytest.raises(ValueError, match="coverage"):
        env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (report,),
            reason="must fail",
        )


def test_unknown_review_is_not_completion_and_teacher_mask_cannot_be_forged(
    env: Environment,
) -> None:
    image = env.page((1,))
    env.apply(
        BallEdits(
            operations=(env.replace(1, image, presence="unknown"),), images=(image,)
        )
    )
    report = env.telemetry((image,))
    with pytest.raises(ValueError, match="no eligible"):
        env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (report,),
            reason="must remain deferred",
        )
    env.apply(BallEdits(operations=(env.replace(1, image, observations=(visible(),)),)))
    adopted = env.adapter.prepare_adoption(
        env.layout, env.context, env.manifest, env.head(), (report,), reason="resolved"
    )
    receipt = read_document(env.layout, adopted, AdoptionReceipt)
    mask = read_document(env.layout, receipt.teacher_mask, BallTeacherMask)
    forged = mask.model_copy(
        update={
            "rows": (
                mask.rows[0].model_copy(update={"heatmap": "negative"}),
                *mask.rows[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="teacher mask"):
        env.store.seal(
            env.head(),
            env.save(receipt.model_copy(update={"teacher_mask": env.save(forged)})),
            env.adapter,
        )


def test_review_policy_has_no_silent_numeric_defaults() -> None:
    with pytest.raises(ValidationError):
        BallReviewPolicy.model_validate_json('{"allow_prediction_confirmation":true}')
