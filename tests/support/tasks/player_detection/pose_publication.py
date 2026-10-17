"""CPU fixture spanning actual bbox review, tracking, ID decisions and publication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_detection.generate_dataset.frame_store.clip import SourceFrame
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    LegacyClipRange,
    LegacySourceGroup,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    freeze_legacy_store,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, FrameRange
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, ProductPlan
from src.tasks.base.annotation.predictions import PredictionReceipt
from src.tasks.person_tracking.features import FeatureExtractor, UnpromptedEncoder
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tasks.player_detection.annotation.contracts import Box, PlayerPredictions
from src.tasks.player_detection.annotation.player_pose.approved import PosePolicy
from src.tasks.player_detection.annotation.player_pose.corrected import (
    BboxInput,
    IdentityReview,
    TrackingModels,
    finalize_review,
    load_tracking,
    pin_input,
    track_corrected,
)
from src.tasks.player_detection.annotation.review.contracts import (
    ConfirmEmpty,
    PlayerEdits,
    PlayerObservation,
    manual_observation_id,
)
from src.tasks.player_detection.generate_dataset.publishing.publisher import (
    PlayerPublicationPolicy,
)
from tests.support.tasks.ball_detection.store import ball, write_store_clip
from tests.support.tasks.player_detection.annotation import Environment


class Encoder:
    name = "synthetic"
    input_size = (16, 8)

    def embed(self, crops):
        return torch.tensor([[1.0, 0.0]]).repeat(len(crops), 1)


class Linker(AFLink):
    def __init__(self):
        pass

    def links(self, boxes, seen):
        return {i: i for i in range(len(boxes))}, []


class PublicationFixture:
    def __init__(
        self,
        root: Path,
        *,
        version="one",
        base: ArtifactRef | None = None,
        video="fixture",
        presence_threshold=0.0,
        all_empty=False,
        finalize=True,
        estimated=False,
    ):
        def products(plan: CampaignPlan) -> CampaignPlan:
            task = plan.tasks[0]
            self.pose_policy = write_document(
                AnnotationLayout.testing(root),
                root / "campaigns" / (version + "-pose-policy.json"),
                PosePolicy(presence_threshold=presence_threshold, chunk_frames=1),
            )
            pose = ProductPlan(
                product="pose",
                dataset_task="ball_detection",
                dataset_version=version,
                component="player_pose",
                base_snapshot=None,
                generation_policy=self.pose_policy,
            )
            return plan.model_copy(
                update={
                    "tasks": (
                        task.model_copy(update={"products": (*task.products, pose)}),
                    )
                }
            )

        self.env = env = Environment(
            root,
            frame_count=5,
            dataset_version=version,
            base_snapshot=base,
            campaign_id=version,
            video_id=video,
            plan_transform=products,
            publication_policy=PlayerPublicationPolicy(
                image_storage="reuse_ball_jpeg", split="inherit_ball_source"
            ),
        )
        prediction = read_document(
            env.layout, env.context.prediction, PredictionReceipt
        )
        raw = read_document(
            env.layout, prediction.prediction.artifact, PlayerPredictions
        )

        def proposed(frame):
            return [
                PlayerObservation(
                    observation_id=o.observation_id,
                    proposal_id=o.observation_id,
                    manual_id=None,
                    classification="player",
                    box_state="observed",
                    box=o.box,
                    origin="model_confirmed",
                    confidence=o.confidence,
                    score_origin="model",
                    duplicate_of=None,
                )
                for o in raw.rows[frame].observations
            ]

        items = proposed(1)
        items[0] = items[0].model_copy(
            update={
                "box": Box(x1=2.0, y1=2.0, x2=25.0, y2=31.0),
                "origin": "manual",
                "score_origin": "missing",
                "confidence": 0.0,
            }
        )
        self.corrected, self.duplicate = items
        for key, coordinates in (
            ("added", (27.0, 2.0, 37.0, 30.0)),
            ("audience", (40.0, 2.0, 60.0, 31.0)),
        ):
            items.append(
                PlayerObservation(
                    observation_id=manual_observation_id(env.manifest.clip_id, 1, key),
                    proposal_id=None,
                    manual_id=key,
                    classification="player",
                    box_state="observed",
                    box=Box(
                        x1=coordinates[0],
                        y1=coordinates[1],
                        x2=coordinates[2],
                        y2=coordinates[3],
                    ),
                    origin="manual",
                    confidence=0.0,
                    score_origin="missing",
                    duplicate_of=None,
                )
            )
        self.added, self.audience = items[2:]
        if estimated:
            items.append(
                PlayerObservation(
                    observation_id=manual_observation_id(
                        env.manifest.clip_id, 1, "estimate"
                    ),
                    proposal_id=None,
                    manual_id="estimate",
                    classification="player",
                    box_state="estimated",
                    box=Box(x1=5.0, y1=2.0, x2=10.0, y2=30.0),
                    origin="manual",
                    confidence=0.0,
                    score_origin="missing",
                    duplicate_of=None,
                )
            )
        image = env.page((1, 2, 3))
        operations = (
            env.replace(
                1,
                image,
                observations=tuple(sorted(items, key=lambda o: o.observation_id)),
            ),
            ConfirmEmpty(scope=FrameRange(start=2, stop=3), evidence=(image.sha256,)),
            env.replace(
                3,
                image,
                observations=tuple(sorted(proposed(3), key=lambda o: o.observation_id)),
                coverage="partial",
            ),
        )
        if all_empty:
            operations = tuple(
                env.replace(
                    frame,
                    image,
                    observations=tuple(
                        sorted(
                            (
                                item.model_copy(
                                    update={
                                        "classification": "non_player",
                                        "box_state": "not_applicable",
                                        "box": None,
                                        "origin": "manual",
                                        "score_origin": "missing",
                                        "confidence": 0.0,
                                    }
                                )
                                for item in proposed(frame)
                            ),
                            key=lambda item: item.observation_id,
                        )
                    ),
                )
                for frame in (1, 2, 3)
            )
        env.apply(PlayerEdits(operations=operations, images=(image,)))
        telemetry = env.telemetry(env.labels().images)
        self.adoption = env.adapter.prepare_adoption(
            env.layout,
            env.context,
            env.manifest,
            env.head(),
            (telemetry,),
            reason="synthetic bbox review",
        )
        directory = env.layout.repository / "data/ball_detection" / (version + "-ball")
        frames = [
            SourceFrame(
                f.clip_pts,
                True,
                1 <= f.frame_index < 4,
                False,
                "unlabeled",
                (ball(xy=(10.0, 10.0)),),
            )
            for f in env.manifest.frames
        ]
        write_store_clip(
            directory,
            env.manifest.clip_id,
            frames,
            source="chat_annotation",
            size=(64, 32),
        )
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["clips"][0].update(
            time_base=env.manifest.time_base,
            media_sha256=env.manifest.media.digest,
            annotation_sha256="a" * 64,
            group_id=env.manifest.source.canonical_id,
        )
        metadata_path.write_text(json.dumps(metadata))
        self.ball_snapshot = freeze_legacy_store(
            env.layout,
            directory,
            (
                LegacySourceGroup(
                    source="chat_annotation",
                    group_id=env.manifest.source.canonical_id,
                    source_id=env.manifest.source.canonical_id,
                ),
            ),
            clip_ranges=(
                LegacyClipRange(
                    store_clip_id=env.manifest.clip_id,
                    source_range=env.manifest.source_range,
                ),
            ),
        )
        self.input = pin_input(
            env.layout,
            BboxInput(adoption=self.adoption, ball_snapshot=self.ball_snapshot),
        )
        self.extractor = FeatureExtractor(None, UnpromptedEncoder(Encoder(), 2))
        self.models = TrackingModels(clip_reid=env.blob, aflink=env.blob)
        self.tracking = track_corrected(
            env.layout,
            self.input,
            models=self.models,
            extractor=self.extractor,
            aflink=Linker(),
        )
        self.review = self.identity_review()
        self.final = finalize_review(env.layout, self.review) if finalize else None

    def identity_review(self, *, status="approved", unknown=False):
        env = self.env
        _, receipt, raw = load_tracking(env.layout, self.tracking)
        segments = []
        for at, track in enumerate(raw["track_ids"]):
            for frame in np.flatnonzero(raw["detection_rows"][at] >= 0):
                identity = str(raw["observation_ids"][at, frame])
                role, player = "non_player", None
                if identity == self.corrected.observation_id:
                    role, player = (
                        ("unknown", None) if unknown else ("player", "player_1")
                    )
                elif identity == self.duplicate.observation_id:
                    role, player = (
                        ("non_player", None) if unknown else ("duplicate", "player_1")
                    )
                elif identity == self.added.observation_id:
                    role, player = "player", "player_2"
                segments.append(
                    dict(
                        raw_track_id=int(track),
                        start_frame=int(frame),
                        stop_frame=int(frame) + 1,
                        role=role,
                        player_id=player,
                        evidence_frames=[int(frame)],
                        reason="synthetic ID evidence",
                    )
                )
        image = env.labels().images[0].image
        decision = dict(
            clip_id=env.manifest.clip_id,
            raw_tracks_sha256=receipt.tracks.digest,
            status=status,
            reviewed_sheets=[Path(image.path).name],
            segments=segments,
            notes="synthetic only",
        )
        payload = json.dumps(decision).encode()
        import hashlib

        decision_ref = write_asset(
            env.layout,
            env.layout.path(
                "annotations",
                "player-pose",
                "decisions",
                hashlib.sha256(payload).hexdigest() + ".json",
            ),
            payload,
        )
        return env.save(
            IdentityReview(
                tracking=self.tracking,
                decision=decision_ref,
                images=(image,),
                required_sheets=(Path(image.path).name,),
            )
        )
