from __future__ import annotations

import base64
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from pydantic import ValidationError

from src.tasks.ball_detection.annotation.contracts import (
    BallPredictions,
    BallProposal,
    MDDInferenceConfig,
)
from src.tasks.ball_detection.annotation.review.adapter import BallReviewAdapter
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
from src.tasks.ball_detection.annotation.validation import validate_predictions
from src.tasks.ball_detection.annotation.windows import window_owners
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
    FrameMap,
    FrameRange,
    ModelRef,
    ReviewEvidence,
    SourceIdentity,
    content_digest,
)
from src.tasks.base.annotation.editing import (
    EditCommand,
    ReviewContext,
    Revision,
    RevisionConflict,
    RevisionStore,
)
from src.tasks.base.annotation.editing.contracts import EditReceipt
from src.tasks.base.annotation.handoff import build_handoff, read_handoff
from src.tasks.base.annotation.images import render_page
from src.tasks.base.annotation.images.contracts import Crop, Point, RenderRequest, Tile
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, save_plan
from src.tasks.base.annotation.predictions import PredictionContext, Predictions
from src.tasks.base.annotation.telemetry.capture import capture
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    Launch,
    PreparedWorker,
    WorkRequest,
)


class ProposalFixture:
    task = "ball"

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
        rows = tuple(
            BallProposal(
                frame_index=f.frame_index,
                source_frame_index=f.source_frame_index,
                source_pts=f.source_pts,
                clip_pts=f.clip_pts,
                x=16.0,
                y=12.0,
                window_start=owner,
            )
            for f, owner in zip(
                manifest.frames, window_owners(len(manifest.frames), 16), strict=True
            )
        )
        reference: ArtifactRef = write_document(
            layout,
            directory / "coordinates.json",
            BallPredictions(context_digest=context.sha256, rows=rows),
        )
        return reference

    def validate(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        result: ArtifactRef,
    ) -> None:
        validate_predictions(layout, context, manifest, result)


class Environment:
    def __init__(self, root: Path, **policy_changes: Any) -> None:
        self.layout = AnnotationLayout.testing(root)
        media_path = self.layout.path("clips", "fixture", "media.mp4")
        media_path.parent.mkdir(parents=True)
        pts, durations, current = [], [], 0
        for i in range(32):
            duration = 3000 if i % 2 == 0 else 6000
            pts.append(current)
            durations.append(duration)
            current += duration
        encode_video(
            media_path,
            width=64,
            height=32,
            time_base=Fraction(1, 90000),
            rate=Fraction(30),
            frames=(
                (
                    av.VideoFrame.from_ndarray(
                        np.full((32, 64, 3), i * 7, dtype=np.uint8), format="rgb24"
                    ),
                    pts[i],
                    durations[i],
                )
                for i in range(32)
            ),
            crf=16,
            preset="ultrafast",
        )
        native = probe_video(media_path)
        media = annotation_reference(self.layout, media_path)
        self.manifest = ClipManifest(
            source=SourceIdentity(
                provider="local",
                channel_id="fixture",
                video_id="fixture",
                media=media,
                origin_url="local:fixture",
            ),
            media=media,
            preparation_digest="a" * 64,
            width=64,
            height=32,
            time_base=str(native.time_base),
            nominal_fps="30",
            source_range=FrameRange(start=100, stop=132),
            target_range=FrameRange(start=1, stop=31),
            frames=tuple(
                FrameMap(
                    frame_index=i,
                    source_frame_index=i + 100,
                    source_pts=p + 90000,
                    clip_pts=p,
                    duration_pts=d,
                )
                for i, (p, d) in enumerate(
                    zip(native.pts, native.durations, strict=True)
                )
            ),
        )
        self.manifest_ref = self.save(self.manifest)
        self.blob = write_asset(
            self.layout,
            self.layout.path("campaigns", "fixture.txt"),
            b"synthetic policy decision and model; no production adoption",
        )
        self.policy = BallReviewPolicy(
            **{
                "allow_prediction_confirmation": True,
                "minimum_reviewed_target_fraction": 0.0,
                "minimum_resolved_target_fraction": 0.0,
                "minimum_image_scale": 0.5,
                "minimum_absence_image_scale": 0.5,
                "maximum_interpolation_seconds": 1.0,
                "unreviewed_teacher": "exclude",
                "supervision": "observed_only",
                **policy_changes,
            }
        )
        policy_ref = self.save(self.policy)
        inference = self.save(
            MDDInferenceConfig(stride=16, device="cpu", precision="float32")
        )
        model = ModelRef(
            task="ball",
            version="fixture",
            training_weights=self.blob,
            inference_weights=self.blob,
            configuration=self.blob,
        )
        ref = self.blob.model_dump()
        self.plan = CampaignPlan.model_validate_json(
            json.dumps(
                {
                    "campaign_id": "review",
                    "phase": "production",
                    "annotation_root": str(root),
                    "counts": {
                        "channels": 1,
                        "videos_per_channel": 1,
                        "max_clips_per_video": 1,
                    },
                    "selection": ref,
                    "preparation": ref,
                    "code_commit": "a" * 40,
                    "code_diff_digest": "b" * 64,
                    "agent_model": "gpt-6.1-sol",
                    "agent_effort": "high",
                    "resources": {
                        "codex_workers": 1,
                        "additional_storage_bytes": 100_000_000,
                        "minimum_free_bytes": 1,
                        "total_pss_bytes": 1024**3,
                        "gpu_seconds": 60,
                        "stages": [
                            {
                                "stage": "review",
                                "kind": "codex",
                                "parallel": 1,
                                "threads_per_process": 1,
                            }
                        ],
                    },
                    "tasks": [
                        {
                            "task": "ball",
                            "model": model.model_dump(),
                            "inference": inference.model_dump(),
                            "review": {
                                "policy": policy_ref.model_dump(),
                                "preview_evidence": ref,
                                "decision": ref,
                            },
                            "learning": None,
                            "products": [
                                {
                                    "product": "frames",
                                    "dataset_task": "ball_detection",
                                    "dataset_version": "fixture",
                                    "component": "frames",
                                    "base_snapshot": None,
                                    "generation_policy": ref,
                                }
                            ],
                        }
                    ],
                }
            )
        )
        self.plan_ref = annotation_reference(
            self.layout, save_plan(self.layout, self.plan)
        )
        predictions = Predictions(self.layout, self.plan)
        with predictions.registry.transaction() as db:
            db.execute(
                "INSERT INTO sources(id,provider,channel_id,video_id,campaign_id,candidate,state) VALUES (?,?,?,?,?,?,?)",
                (
                    self.manifest.source.canonical_id,
                    "local",
                    "fixture",
                    "fixture",
                    "review",
                    "{}",
                    "prepared",
                ),
            )
            db.execute(
                "INSERT INTO clips VALUES (?,?,?,?,?)",
                (
                    self.manifest.clip_id,
                    self.manifest.source.canonical_id,
                    "review",
                    self.manifest_ref.canonical_bytes().decode(),
                    media.digest,
                ),
            )
            db.execute(
                "INSERT INTO annotation_tasks VALUES (?,?,?,?,?,?,?)",
                (
                    content_digest(["ball", self.manifest.clip_id]),
                    "review",
                    "ball",
                    self.manifest.clip_id,
                    model.sha256,
                    inference.digest,
                    "pending",
                ),
            )
        prediction = predictions.run(self.manifest_ref, ProposalFixture())
        self.context = ReviewContext(
            plan=self.plan_ref,
            task="ball",
            manifest=self.manifest_ref,
            prediction=prediction,
        )
        self.adapter, self.store = BallReviewAdapter(), RevisionStore(self.layout)
        self.initial = self.store.initialize(self.context, self.adapter)
        self.context_ref = read_document(self.layout, self.initial, Revision).context
        self.actor = self.new_actor("first")

    def save(self, document: Any) -> ArtifactRef:
        result: ArtifactRef = write_document(
            self.layout,
            self.layout.path("campaigns", "fixture", document.sha256 + ".json"),
            document,
        )
        return result

    def new_actor(self, attempt: str) -> ArtifactRef:
        directory = self.layout.path("campaigns", "review", "workers", attempt)
        request = WorkRequest(
            adapter="ball_review",
            adapter_version="v1",
            stage="review",
            subject_id=self.manifest.clip_id,
            payload=self.context_ref,
            inputs=(self.context.manifest, self.context.prediction),
        )
        launch = Launch(
            work_id=request.work_id,
            attempt=attempt,
            plan=self.plan_ref,
            request=request,
            prepared=PreparedWorker(prompt=self.blob),
            directory=str(directory),
            code_root=str(self.layout.repository),
            codex_binary="codex",
            codex_home=str(self.layout.repository / "codex"),
            timeout_seconds=60.0,
            test_root=str(self.layout.root),
        )
        result: ArtifactRef = write_document(
            self.layout, directory / "launch.json", launch
        )
        return result

    def head(self) -> ArtifactRef:
        result: ArtifactRef = self.store.current(self.context_ref, self.adapter)[0]
        return result

    def labels(self, reference: ArtifactRef | None = None) -> BallLabels:
        revision = read_document(self.layout, reference or self.head(), Revision)
        result: BallLabels = read_document(self.layout, revision.payload, BallLabels)
        return result

    def page(
        self,
        indices: tuple[int, ...],
        *,
        crop: Crop | None = None,
        revision: ArtifactRef | None = None,
        markers: tuple[Point, ...] = (),
    ) -> RenderedImage:
        result: RenderedImage = render_page(
            self.layout,
            RenderRequest(
                manifest=self.context.manifest,
                annotation=revision or self.context.prediction,
                annotation_kind="revision" if revision else "prediction",
                drawing=("revision" if revision else "prediction")
                if markers
                else "plain",
                reason="synthetic test image",
                tiles=tuple(
                    Tile(
                        frame_index=i,
                        crop=crop or Crop(left=0, top=0, right=64, bottom=32),
                        drawings=markers,
                    )
                    for i in indices
                ),
                columns=min(4, len(indices)),
                tile_width=64,
                tile_height=32,
            ),
        )
        return result

    def command(
        self, edits: BallEdits, *, actor: ArtifactRef | None = None
    ) -> EditCommand:
        return EditCommand(
            context=self.context_ref,
            expected_revision=self.head(),
            changes=self.save(edits),
            actor=actor or self.actor,
            reason="explicit synthetic review edit",
        )

    def apply(self, edits: BallEdits) -> ArtifactRef:
        result: ArtifactRef = self.store.apply(self.command(edits), self.adapter)
        return result

    def replace(
        self,
        index: int,
        image: RenderedImage,
        *,
        observations: tuple[BallObservation, ...] = (),
        presence: str = "present",
        notes: tuple[Note, ...] = (),
        boundaries: tuple[str, ...] = (),
    ) -> ReplaceFrame:
        return ReplaceFrame(
            frame_index=index,
            expected_frame_digest=self.labels().frames[index].sha256,
            decision=FrameDecision(
                reviewed=True,
                presence=presence,
                observations=observations,
                evidence=(image.sha256,),
                notes=notes,
                boundaries=boundaries,
            ),
        )

    def telemetry(
        self,
        catalog: tuple[RenderedImage, ...],
        *,
        shown: tuple[RenderedImage, ...] | None = None,
        actor: ArtifactRef | None = None,
    ) -> ArtifactRef:
        actor = actor or self.actor
        launch = read_document(self.layout, actor, Launch)
        execution = ExitReceipt(
            launch_digest=actor.digest,
            exit_code=0,
            status="succeeded",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:01Z",
            elapsed_seconds=1.0,
            events=self.blob,
            errors=self.blob,
            thread_id="11111111-1111-1111-1111-111111111111",
            reason=None,
        )
        execution_ref = write_document(
            self.layout, Path(launch.directory) / "exit.json", execution
        )

        def event(kind: str, **payload: Any) -> dict:
            return {
                "type": kind,
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": payload,
            }

        events = [
            event(
                "session_meta",
                id=execution.thread_id,
                cwd=str(Path(launch.directory) / "work"),
                cli_version="fixture",
                model_provider="openai",
            ),
            event("turn_context", model=self.plan.agent_model),
        ]
        for i, image in enumerate(catalog if shown is None else shown):
            events.extend(
                [
                    event(
                        "response_item",
                        type="custom_tool_call",
                        call_id=f"view-{i}",
                        name="view_image",
                        input=json.dumps(
                            {"path": str(self.layout.artifact_path(image.image))}
                        ),
                    ),
                    event(
                        "response_item",
                        type="custom_tool_call_output",
                        call_id=f"view-{i}",
                        output=[
                            {
                                "type": "input_image",
                                "detail": "original",
                                "image_url": "data:image/jpeg;base64,"
                                + base64.b64encode(
                                    self.layout.artifact_path(image.image).read_bytes()
                                ).decode(),
                            }
                        ],
                    ),
                ]
            )
        events.append(
            event(
                "token_usage_record",
                response_id="response-1",
                usage={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            )
        )
        rollout = Path(launch.directory) / "rollout.jsonl"
        rollout.write_text("".join(json.dumps(e) + "\n" for e in events))
        result: ArtifactRef = capture(
            self.layout, actor, execution_ref, catalog, rollout=rollout
        )
        return result


@pytest.fixture
def env(tmp_path: Path) -> Environment:
    return Environment(tmp_path / "annotation")


def visible(x: float = 16.0, *, track: str = "ball") -> BallObservation:
    return BallObservation(
        track_id=track, state="visible", center=(x, 12.0), origin="manual"
    )


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
