from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import AdoptionReceipt, FrameRange
from src.tasks.base.annotation.editing import RevisionConflict
from src.tasks.base.annotation.editing.contracts import EditReceipt
from src.tasks.base.annotation.images import render_page
from src.tasks.base.annotation.images.contracts import Box as DrawingBox
from src.tasks.base.annotation.images.contracts import Crop, RenderRequest, Tile
from src.tasks.base.annotation.predictions import PredictionReceipt
from src.tasks.player_detection.annotation.contracts import Box, PlayerPredictions
from src.tasks.player_detection.annotation.review.contracts import (
    ConfirmEmpty,
    ConfirmPredictions,
    Note,
    PlayerEdits,
    PlayerObservation,
    PlayerReviewPolicy,
    PlayerTeacherMask,
    manual_observation_id,
)
from src.tasks.player_detection.annotation.review.validation import teacher_rows
from tests.support.tasks.player_detection.annotation import Environment


@pytest.fixture
def env(tmp_path: Path) -> Environment:
    return Environment(tmp_path / "annotation")


def proposals(env: Environment, index: int = 1) -> tuple[PlayerObservation, ...]:
    receipt = read_document(env.layout, env.context.prediction, PredictionReceipt)
    rows = read_document(env.layout, receipt.prediction.artifact, PlayerPredictions)
    return tuple(
        sorted(
            (
                PlayerObservation(
                    observation_id=p.observation_id,
                    proposal_id=p.observation_id,
                    manual_id=None,
                    classification="player",
                    box_state="observed",
                    box=p.box,
                    origin="model_confirmed",
                    confidence=p.confidence,
                    score_origin="model",
                    duplicate_of=None,
                )
                for p in rows.rows[index].observations
            ),
            key=lambda o: o.observation_id,
        )
    )


def manual(item: PlayerObservation, **changes: object) -> PlayerObservation:
    data = {
        **item.model_dump(mode="json"),
        "origin": "manual",
        "confidence": 0.0,
        "score_origin": "missing",
        **changes,
    }
    import json

    result: PlayerObservation = PlayerObservation.model_validate_json(json.dumps(data))
    return result


def reject(item: PlayerObservation) -> PlayerObservation:
    return manual(
        item, classification="non_player", box_state="not_applicable", box=None
    )


def confirm(env: Environment, index: int = 1) -> None:
    image = env.page((index,))
    env.apply(
        PlayerEdits(
            operations=(
                ConfirmPredictions(
                    scope=FrameRange(start=index, stop=index + 1),
                    evidence=(image.sha256,),
                ),
            ),
            images=(image,),
        )
    )


def test_policy_requires_all_explicit_values_and_definitions(env: Environment) -> None:
    for key in (
        "target_definition",
        "bbox_definition",
        "allow_prediction_confirmation",
        "minimum_reviewed_target_fraction",
        "minimum_resolved_target_fraction",
        "minimum_image_scale",
        "minimum_full_frame_image_scale",
        "supervision",
    ):
        data = env.policy.model_dump(mode="json")
        del data[key]
        with pytest.raises(ValidationError):
            PlayerReviewPolicy.model_validate(data)
    with pytest.raises(ValidationError, match="explicit"):
        PlayerReviewPolicy.model_validate(
            {**env.policy.model_dump(), "bbox_definition": " "}
        )


def test_initial_native_frames_and_empty_predictions_are_not_teachers(
    env: Environment,
) -> None:
    labels = env.labels()
    assert tuple(f.frame for f in labels.frames) == env.manifest.frames
    assert all(
        not f.decision.reviewed and not f.decision.observations for f in labels.frames
    )
    assert all(row.supervision == "ignore" for row in teacher_rows(labels))
    assert env.store.initialize(env.context, env.adapter) == env.initial


@pytest.mark.parametrize("change", ["pts", "target", "bounds", "omission"])
def test_native_identity_bounds_and_candidate_inventory_cannot_drift(
    env: Environment, change: str
) -> None:
    image = env.page((1,))
    items = proposals(env)
    if change == "bounds":
        items = (
            manual(items[0], box={"x1": 1.0, "y1": 1.0, "x2": 65.0, "y2": 32.0}),
            items[1],
        )
    if change == "omission":
        items = items[:1]
    if change in {"bounds", "omission"}:
        with pytest.raises(ValueError, match="native image edges|every original"):
            env.apply(
                PlayerEdits(
                    operations=(env.replace(1, image, observations=items),),
                    images=(image,),
                )
            )
    else:
        labels = env.labels()
        rows = list(labels.frames)
        row = rows[1]
        rows[1] = row.model_copy(
            update={
                "frame": row.frame.model_copy(
                    update={"source_pts": row.frame.source_pts + 1}
                )
            }
            if change == "pts"
            else {"target": False}
        )
        with pytest.raises(ValueError, match="native frame/PTS"):
            env.adapter.validate(
                env.layout,
                env.context,
                env.manifest,
                labels.model_copy(update={"frames": tuple(rows)}),
            )
    assert env.head() == env.initial


def test_confirmed_proposals_keep_identity_confidence_and_immutable_raw(
    env: Environment,
) -> None:
    raw = env.layout.artifact_path(env.context.prediction).read_bytes()
    image = env.page((1,))
    command = env.command(
        PlayerEdits(
            operations=(
                ConfirmPredictions(
                    scope=FrameRange(start=1, stop=2), evidence=(image.sha256,)
                ),
            ),
            images=(image,),
        )
    )
    receipt = env.store.apply(command, env.adapter)
    assert env.store.apply(command, env.adapter) == receipt
    assert env.labels().frames[1].decision.observations == proposals(env)
    assert env.labels().frames[1].review_actor == env.actor
    assert env.layout.artifact_path(env.context.prediction).read_bytes() == raw
    assert teacher_rows(env.labels())[1].supervision == "positive"
    assert read_document(env.layout, receipt, EditReceipt).summary.changed_rows == 1


def test_correction_addition_rejection_keep_stable_ids_and_missing_scores(
    env: Environment,
) -> None:
    image = env.page((1,))
    original = proposals(env)
    corrected = manual(original[0], box={"x1": 1.0, "y1": 1.0, "x2": 22.0, "y2": 31.0})
    added = PlayerObservation(
        observation_id=manual_observation_id(env.manifest.clip_id, 1, "missed"),
        proposal_id=None,
        manual_id="missed",
        classification="player",
        box_state="observed",
        box=Box(x1=40.0, y1=2.0, x2=64.0, y2=32.0),
        origin="manual",
        confidence=0.0,
        score_origin="missing",
        duplicate_of=None,
    )
    observations = tuple(
        sorted((corrected, reject(original[1]), added), key=lambda o: o.observation_id)
    )
    env.apply(
        PlayerEdits(
            operations=(env.replace(1, image, observations=observations),),
            images=(image,),
        )
    )
    fixed = env.head()
    assert set(teacher_rows(env.labels())[1].observation_ids) == {
        corrected.observation_id,
        added.observation_id,
    }
    assert all(o.score_origin == "missing" and o.confidence == 0 for o in observations)
    rejected = tuple(reject(o) for o in observations)
    env.apply(PlayerEdits(operations=(env.replace(1, image, observations=rejected),)))
    assert teacher_rows(env.labels())[1].supervision == "negative"
    assert env.labels(fixed).frames[1].decision.observations == observations
    with pytest.raises(ValueError, match="retain observation IDs"):
        env.apply(
            PlayerEdits(operations=(env.replace(1, image, observations=rejected[:-1]),))
        )


@pytest.mark.parametrize(
    "change", ["score", "model_box", "foreign_candidate", "manual_id"]
)
def test_score_and_identity_forgery_are_rejected(env: Environment, change: str) -> None:
    image = env.page((1,))
    items = list(proposals(env))
    with pytest.raises(ValueError):
        if change == "score":
            items[0] = manual(items[0], confidence=0.9)
        elif change == "model_box":
            items[0] = items[0].model_copy(
                update={"box": Box(x1=1.0, y1=1.0, x2=20.0, y2=30.0)}
            )
        elif change == "foreign_candidate":
            items[0] = proposals(env, 3)[0]
        else:
            items.append(manual(items[0], proposal_id=None, manual_id="new"))
        env.apply(
            PlayerEdits(
                operations=(
                    env.replace(
                        1,
                        image,
                        observations=tuple(
                            sorted(items, key=lambda o: o.observation_id)
                        ),
                    ),
                ),
                images=(image,),
            )
        )
    assert env.head() == env.initial


def test_duplicate_references_retained_player_and_every_candidate_is_accounted(
    env: Environment,
) -> None:
    image = env.page((1,))
    a, b = proposals(env)
    duplicate = manual(
        b,
        classification="duplicate",
        box_state="not_applicable",
        box=None,
        duplicate_of=a.observation_id,
    )
    env.apply(
        PlayerEdits(
            operations=(env.replace(1, image, observations=(a, duplicate)),),
            images=(image,),
        )
    )
    assert teacher_rows(env.labels())[1].observation_ids == (a.observation_id,)
    with pytest.raises(ValueError, match="retained player"):
        env.replace(1, image, observations=(reject(a), duplicate))


def test_only_explicit_empty_review_creates_negative_and_bulk_is_atomic(
    env: Environment,
) -> None:
    image = env.page((1, 2))
    with pytest.raises(ValueError, match="empty predictions"):
        env.apply(
            PlayerEdits(
                operations=(
                    ConfirmPredictions(
                        scope=FrameRange(start=1, stop=3), evidence=(image.sha256,)
                    ),
                ),
                images=(image,),
            )
        )
    assert env.head() == env.initial
    with pytest.raises(ValueError, match="explicit candidate rejection"):
        env.apply(
            PlayerEdits(
                operations=(
                    ConfirmEmpty(
                        scope=FrameRange(start=1, stop=2), evidence=(image.sha256,)
                    ),
                ),
                images=(image,),
            )
        )
    env.apply(
        PlayerEdits(
            operations=(
                ConfirmEmpty(
                    scope=FrameRange(start=2, stop=3), evidence=(image.sha256,)
                ),
            ),
            images=(image,),
        )
    )
    assert teacher_rows(env.labels())[2].supervision == "negative"


@pytest.mark.parametrize("kind", ["partial", "estimated", "unresolved", "blocking"])
def test_incomplete_estimated_or_unresolved_inventory_excludes_whole_frame(
    env: Environment, kind: str
) -> None:
    image = env.page((1,))
    items = list(proposals(env))
    if kind == "estimated":
        items[0] = manual(items[0], box_state="estimated")
    if kind == "unresolved":
        items[0] = manual(items[0], box_state="unresolved", box=None)
    note = (
        (Note(note_id="uncertain", text="unclear target", blocking=True),)
        if kind == "blocking"
        else ()
    )
    env.apply(
        PlayerEdits(
            operations=(
                env.replace(
                    1,
                    image,
                    observations=tuple(items),
                    coverage="partial" if kind == "partial" else "complete",
                    notes=note,
                ),
            ),
            images=(image,),
        )
    )
    row = teacher_rows(env.labels())[1]
    assert row.supervision == "ignore" and row.observation_ids == ()
    with pytest.raises(ValueError, match="no eligible"):
        env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (env.telemetry((image,)),),
            reason="fixture",
        )


def test_full_frame_and_bbox_detail_have_independent_scale_limits(
    tmp_path: Path,
) -> None:
    env = Environment(
        tmp_path / "annotation",
        minimum_image_scale=1.0,
        minimum_full_frame_image_scale=0.5,
    )
    image = render_page(
        env.layout,
        RenderRequest(
            manifest=env.context.manifest,
            annotation=env.context.prediction,
            annotation_kind="prediction",
            drawing="plain",
            reason="half size full frame",
            tiles=(Tile(frame_index=2, crop=Crop(left=0, top=0, right=64, bottom=32)),),
            columns=1,
            tile_width=32,
            tile_height=32,
        ),
    )
    env.apply(
        PlayerEdits(
            operations=(
                ConfirmEmpty(
                    scope=FrameRange(start=2, stop=3), evidence=(image.sha256,)
                ),
            ),
            images=(image,),
        )
    )
    assert teacher_rows(env.labels())[2].supervision == "negative"
    crop = env.page((1,), crop=Crop(left=0, top=0, right=63, bottom=32))
    with pytest.raises(ValueError, match="full-frame"):
        env.apply(
            PlayerEdits(
                operations=(env.replace(1, crop, observations=proposals(env)),),
                images=(crop,),
            )
        )


def test_cropped_bbox_or_forged_overlay_is_not_image_evidence(env: Environment) -> None:
    crop = env.page((1,), crop=Crop(left=0, top=0, right=20, bottom=32))
    with pytest.raises(ValueError, match="fully contained"):
        env.apply(
            PlayerEdits(
                operations=(
                    env.replace(
                        1, crop, observations=proposals(env), coverage="partial"
                    ),
                ),
                images=(crop,),
            )
        )
    forged = env.page(
        (1,), markers=(DrawingBox(left=1.0, top=1.0, right=10.0, bottom=10.0),)
    )
    with pytest.raises(ValueError, match="overlay"):
        env.apply(
            PlayerEdits(
                operations=(env.replace(1, forged, observations=proposals(env)),),
                images=(forged,),
            )
        )


def test_confirmation_requires_each_frame_image_and_policy_permission(
    env: Environment, tmp_path: Path
) -> None:
    image = env.page((1,))
    with pytest.raises(ValueError, match="image references"):
        env.apply(
            PlayerEdits(
                operations=(
                    ConfirmPredictions(
                        scope=FrameRange(start=3, stop=4), evidence=(image.sha256,)
                    ),
                ),
                images=(image,),
            )
        )
    other = Environment(tmp_path / "disabled", allow_prediction_confirmation=False)
    with pytest.raises(ValueError, match="disabled"):
        confirm(other)


def test_adoption_requires_own_actual_images_then_seals_exact_head(
    env: Environment,
) -> None:
    confirm(env)
    image = env.labels().images[0]
    other = env.new_actor("other")
    wrong = env.telemetry((image,), actor=other)
    with pytest.raises(ValueError, match="actually shown"):
        env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (wrong,),
            reason="wrong reviewer",
        )
    proof = env.telemetry((image,))
    adopted = env.adapter.prepare_adoption(
        env.layout,
        env.context,
        env.manifest,
        env.head(),
        (proof,),
        reason="verified fixture",
    )
    receipt = read_document(env.layout, adopted, AdoptionReceipt)
    mask = read_document(env.layout, receipt.teacher_mask, PlayerTeacherMask)
    assert mask.rows == teacher_rows(env.labels())
    env.store.seal(env.head(), adopted, env.adapter)
    assert env.store.seal(env.head(), adopted, env.adapter) == adopted
    with pytest.raises(RevisionConflict, match="sealed"):
        env.apply(
            PlayerEdits(
                operations=(
                    env.replace(
                        1, image, observations=tuple(reject(o) for o in proposals(env))
                    ),
                )
            )
        )


def test_generated_but_unseen_images_cannot_be_adopted(env: Environment) -> None:
    confirm(env)
    with pytest.raises(ValueError, match="actually shown"):
        env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (env.telemetry(env.labels().images, shown=()),),
            reason="not seen",
        )


def test_explicit_coverage_threshold_applies_without_default_cutoff(
    tmp_path: Path,
) -> None:
    env = Environment(tmp_path / "annotation", minimum_reviewed_target_fraction=1.0)
    confirm(env)
    with pytest.raises(ValueError, match="reviewed target coverage"):
        env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (env.telemetry(env.labels().images),),
            reason="insufficient coverage",
        )
