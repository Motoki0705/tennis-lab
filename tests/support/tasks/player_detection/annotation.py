"""Synthetic, CPU-only player review environment with real renderer evidence."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import av
import numpy as np

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    FrameMap,
    FrameRange,
    ModelRef,
    SourceIdentity,
    content_digest,
)
from src.tasks.base.annotation.editing import (
    EditCommand,
    ReviewContext,
    Revision,
    RevisionStore,
)
from src.tasks.base.annotation.images import render_page
from src.tasks.base.annotation.images.contracts import (
    Crop,
    Primitive,
    RenderRequest,
    Tile,
)
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, LearningPlan, save_plan
from src.tasks.base.annotation.predictions import PredictionContext, Predictions
from src.tasks.base.annotation.telemetry.capture import capture
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    Launch,
    PreparedWorker,
    WorkRequest,
)
from src.tasks.player_detection.annotation.contracts import (
    Box,
    DinoInferenceConfig,
    PlayerFrameProposals,
    PlayerPredictions,
    PlayerProposal,
)
from src.tasks.player_detection.annotation.inference import (
    PlayerPredictionAdapter,
    observation_id,
)
from src.tasks.player_detection.annotation.review.adapter import PlayerReviewAdapter
from src.tasks.player_detection.annotation.review.contracts import (
    WORKER_ADAPTER,
    WORKER_VERSION,
    FrameDecision,
    Note,
    PlayerEdits,
    PlayerLabels,
    PlayerObservation,
    PlayerReviewPolicy,
    ReplaceFrame,
)


class ProposalFixture:
    task = "player"

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
        rows = tuple(
            PlayerFrameProposals(
                frame_index=f.frame_index,
                source_frame_index=f.source_frame_index,
                source_pts=f.source_pts,
                clip_pts=f.clip_pts,
                observations=()
                if f.frame_index == 2
                else tuple(
                    PlayerProposal(
                        observation_id=observation_id(context, f.frame_index, rank),
                        candidate_index=rank,
                        box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=score,
                    )
                    for rank, (x1, y1, x2, y2, score) in enumerate(
                        ((4.0, 3.0, 24.0, 30.0, 0.9), (38.0, 5.0, 60.0, 32.0, 0.8))
                    )
                ),
            )
            for f in manifest.frames
        )
        reference: ArtifactRef = write_document(
            layout,
            directory / "boxes.json",
            PlayerPredictions(context_digest=context.sha256, rows=rows),
        )
        return reference

    def validate(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        result: ArtifactRef,
    ) -> None:
        PlayerPredictionAdapter().validate(layout, context, manifest, result)


class Environment:
    def __init__(
        self,
        root: Path,
        *,
        publication_policy: Any = None,
        base_snapshot: ArtifactRef | None = None,
        dataset_version: str = "fixture",
        video_id: str = "fixture",
        campaign_id: str = "review",
        frame_count: int = 32,
        model: ModelRef | None = None,
        learning: LearningPlan | None = None,
        plan_transform: Callable[[CampaignPlan], CampaignPlan] | None = None,
        **policy_changes: Any,
    ) -> None:
        self.layout = AnnotationLayout.testing(root)
        media_path = self.layout.path("clips", video_id, "media.mp4")
        media_path.parent.mkdir(parents=True)
        pts, durations, current = [], [], 0
        for i in range(frame_count):
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
                        np.full(
                            (32, 64, 3),
                            i * 7 + (0 if video_id == "fixture" else 1),
                            dtype=np.uint8,
                        ),
                        format="rgb24",
                    ),
                    pts[i],
                    durations[i],
                )
                for i in range(frame_count)
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
                video_id=video_id,
                media=media,
                origin_url="local:fixture",
            ),
            media=media,
            preparation_digest="a" * 64,
            width=64,
            height=32,
            time_base=str(native.time_base),
            nominal_fps="30",
            source_range=FrameRange(start=100, stop=100 + frame_count),
            target_range=FrameRange(start=1, stop=frame_count - 1),
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
        self.policy = PlayerReviewPolicy(
            **{
                "allow_prediction_confirmation": True,
                "minimum_reviewed_target_fraction": 0.0,
                "minimum_resolved_target_fraction": 0.0,
                "minimum_image_scale": 0.5,
                "minimum_full_frame_image_scale": 0.5,
                "target_definition": "Synthetic colored subjects only",
                "bbox_definition": "Synthetic visible extent clipped to native edges",
                "unreviewed_teacher": "exclude",
                "supervision": "observed_only",
                **policy_changes,
            }
        )
        policy_ref = self.save(self.policy)
        inference = self.save(
            DinoInferenceConfig(
                confidence_threshold=0.5, short_side=32, max_long_side=64
            )
        )
        model = model or ModelRef(
            task="player",
            version="fixture",
            training_weights=self.blob,
            inference_weights=self.blob,
            configuration=self.blob,
        )
        ref = self.blob.model_dump()
        self.plan = CampaignPlan.model_validate_json(
            json.dumps(
                {
                    "campaign_id": campaign_id,
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
                    "agent_effort": "max",
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
                            "task": "player",
                            "model": model.model_dump(),
                            "inference": inference.model_dump(),
                            "review": {
                                "policy": policy_ref.model_dump(),
                                "preview_evidence": ref,
                                "decision": ref,
                            },
                            "learning": learning.model_dump() if learning else None,
                            "products": [
                                {
                                    "product": "boxes",
                                    "dataset_task": "player_detection",
                                    "dataset_version": dataset_version,
                                    "component": "boxes",
                                    "base_snapshot": base_snapshot.model_dump()
                                    if base_snapshot
                                    else None,
                                    "generation_policy": self.save(
                                        publication_policy
                                    ).model_dump()
                                    if publication_policy
                                    else ref,
                                }
                            ],
                        }
                    ],
                }
            )
        )
        if plan_transform is not None:
            self.plan = plan_transform(self.plan)
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
                    video_id,
                    campaign_id,
                    "{}",
                    "prepared",
                ),
            )
            db.execute(
                "INSERT INTO clips VALUES (?,?,?,?,?)",
                (
                    self.manifest.clip_id,
                    self.manifest.source.canonical_id,
                    campaign_id,
                    self.manifest_ref.canonical_bytes().decode(),
                    media.digest,
                ),
            )
            db.execute(
                "INSERT INTO annotation_tasks VALUES (?,?,?,?,?,?,?)",
                (
                    content_digest(["player", self.manifest.clip_id]),
                    campaign_id,
                    "player",
                    self.manifest.clip_id,
                    model.sha256,
                    inference.digest,
                    "pending",
                ),
            )
        prediction = predictions.run(self.manifest_ref, ProposalFixture())
        self.context = ReviewContext(
            plan=self.plan_ref,
            task="player",
            manifest=self.manifest_ref,
            prediction=prediction,
        )
        self.adapter, self.store = PlayerReviewAdapter(), RevisionStore(self.layout)
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
        directory = self.layout.path(
            "campaigns", self.plan.campaign_id, "workers", attempt
        )
        request = WorkRequest(
            adapter=WORKER_ADAPTER,
            adapter_version=WORKER_VERSION,
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

    def labels(self, reference: ArtifactRef | None = None) -> PlayerLabels:
        revision = read_document(self.layout, reference or self.head(), Revision)
        result: PlayerLabels = read_document(
            self.layout, revision.payload, PlayerLabels
        )
        return result

    def page(
        self,
        indices: tuple[int, ...],
        *,
        crop: Crop | None = None,
        revision: ArtifactRef | None = None,
        markers: tuple[Primitive, ...] = (),
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
        self, edits: PlayerEdits, *, actor: ArtifactRef | None = None
    ) -> EditCommand:
        return EditCommand(
            context=self.context_ref,
            expected_revision=self.head(),
            changes=self.save(edits),
            actor=actor or self.actor,
            reason="explicit synthetic review edit",
        )

    def apply(self, edits: PlayerEdits) -> ArtifactRef:
        result: ArtifactRef = self.store.apply(self.command(edits), self.adapter)
        return result

    def replace(
        self,
        index: int,
        image: RenderedImage,
        *,
        observations: tuple[PlayerObservation, ...] = (),
        coverage: Literal["unknown", "partial", "complete"] = "complete",
        notes: tuple[Note, ...] = (),
    ) -> ReplaceFrame:
        return ReplaceFrame(
            frame_index=index,
            expected_frame_digest=self.labels().frames[index].sha256,
            decision=FrameDecision(
                reviewed=True,
                coverage=coverage,
                observations=observations,
                evidence=(image.sha256,),
                notes=notes,
            ),
        )

    def telemetry(
        self,
        catalog: tuple[RenderedImage, ...],
        *,
        shown: tuple[RenderedImage, ...] | None = None,
        actor: ArtifactRef | None = None,
        exit_status: Literal[
            "succeeded", "failed", "quota", "timed_out", "cancelled"
        ] = "succeeded",
        telemetry_fault: Literal["missing_model", "conflicting_response_id"]
        | None = None,
    ) -> ArtifactRef:
        actor = actor or self.actor
        launch = read_document(self.layout, actor, Launch)
        execution = ExitReceipt(
            launch_digest=actor.digest,
            exit_code=0 if exit_status == "succeeded" else 1,
            status=exit_status,
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:01Z",
            elapsed_seconds=1.0,
            events=self.blob,
            errors=self.blob,
            thread_id="11111111-1111-1111-1111-111111111111",
            reason=None
            if exit_status == "succeeded"
            else "synthetic interrupted attempt",
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
        ]
        if telemetry_fault != "missing_model":
            events.append(event("turn_context", model=self.plan.agent_model))
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
        if telemetry_fault == "conflicting_response_id":
            events.append(
                event(
                    "token_usage_record",
                    response_id="response-1",
                    usage={
                        "input_tokens": 200,
                        "output_tokens": 20,
                        "total_tokens": 220,
                    },
                )
            )
        rollout = Path(launch.directory) / "rollout.jsonl"
        rollout.write_text("".join(json.dumps(e) + "\n" for e in events))
        result: ArtifactRef = capture(
            self.layout, actor, execution_ref, catalog, rollout=rollout
        )
        return result
