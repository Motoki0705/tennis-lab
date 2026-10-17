"""Pose manifests publish only completed, identity-checked results or reasoned exclusions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    PublicationReceipt,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import load_plan
from src.tasks.base.annotation.publication import publish_directory
from src.tasks.base.annotation.storage.catalog import StorageCatalog
from src.tasks.base.annotation.storage.references import protection

from .approved import validate_result
from .corrected import load_final


class PosePublicationRequest(Document):
    schema_version: Literal["player.pose-publication.v1"] = "player.pose-publication.v1"
    plan: ArtifactRef
    ball_snapshot: ArtifactRef
    results: tuple[ArtifactRef, ...]


class PublishedPose(Document):
    clip_id: str
    store_clip_id: str
    result: ArtifactRef


class PoseSnapshot(Document):
    schema_version: Literal["player.pose-snapshot.v1"] = "player.pose-snapshot.v1"
    request: ArtifactRef
    ball_snapshot: ArtifactRef
    clips: tuple[PublishedPose, ...]


def validate_pose_snapshot(layout: AnnotationLayout, snapshot: PoseSnapshot) -> None:
    ball, store = load_snapshot(layout, snapshot.ball_snapshot)
    if len(snapshot.clips) != len(ball.clips) or len(
        {c.clip_id for c in snapshot.clips}
    ) != len(snapshot.clips):
        raise ValueError(
            "Pose snapshot must account for every ball snapshot clip exactly once"
        )
    for published, identity in zip(snapshot.clips, ball.clips, strict=True):
        expected_id = identity.canonical_clip_id or identity.clip_id
        result, arrays = validate_result(layout, published.result)
        if (
            published.clip_id,
            published.store_clip_id,
            result.identity.clip_id,
            result.identity.image,
        ) != (expected_id, identity.store_clip_id, expected_id, identity.shard.digest):
            raise ValueError("Pose publication changed clip or image identity")
        clip = store.clip_by_id(identity.store_clip_id)
        inputs, _, _ = load_final(layout, result.final_review)
        if (clip.media_sha256, clip.width, clip.height, clip.time_base, clip.fps) != (
            inputs.clip.media_sha256,
            inputs.clip.width,
            inputs.clip.height,
            inputs.clip.time_base,
            inputs.clip.fps,
        ):
            raise ValueError("Pose publication changed media/geometry identity")
        rows, previous = store.clip_rows(clip), inputs.store.clip_rows(inputs.clip)
        for name in ("frame_index", "pts", "is_target", "annotated"):
            if not np.array_equal(
                store.frames[name][rows], inputs.store.frames[name][previous]
            ):
                raise ValueError(
                    "Pose publication changed native frames/PTS/annotation"
                )
        if arrays is not None and not np.array_equal(
            arrays["pts"], store.frames["pts"][rows]
        ):
            raise ValueError("Published pose PTS differs from reader images")


class PosePublisher:
    def __init__(self, layout: AnnotationLayout) -> None:
        if layout.review_namespace is not None:
            raise ValueError("Benchmark reviews cannot publish poses")
        self.layout = layout

    def publish(self, request: PosePublicationRequest) -> ArtifactRef:
        layout = self.layout
        plan = load_plan(verify_reference(layout, request.plan))
        task = next((t for t in plan.tasks if t.task == "player"), None)
        product = (
            next((p for p in task.products if p.product == "pose"), None)
            if task
            else None
        )
        if (
            plan.phase != "production"
            or plan.annotation_root != str(layout.root)
            or product is None
            or (product.dataset_task, product.component)
            != ("ball_detection", "player_pose")
        ):
            raise ValueError(
                "Pose publication requires the planned ball_detection/player_pose product"
            )
        ball, _ = load_snapshot(layout, request.ball_snapshot)
        results: dict[str, ArtifactRef] = {}
        base = (
            read_document(layout, product.base_snapshot, PoseSnapshot)
            if product.base_snapshot
            else None
        )
        if base:
            validate_pose_snapshot(layout, base)
            results.update({c.clip_id: c.result for c in base.clips})
        added = []
        documents: list[Document] = [request]
        adoptions = []
        for reference in request.results:
            result, _ = validate_result(layout, reference)
            inputs, final, _ = load_final(layout, result.final_review)
            if (
                inputs.context.plan != request.plan
                or result.identity.clip_id in results
                or result.policy != product.generation_policy
            ):
                raise ValueError(
                    "Pose addition duplicates a clip or belongs to another plan"
                )
            results[result.identity.clip_id] = reference
            added.append(result.identity.clip_id)
            adoptions.append(inputs.binding.bbox_review)
            documents.extend((result, final, inputs.input))
        expected_ids = {c.canonical_clip_id or c.clip_id for c in ball.clips}
        if set(results) != expected_ids:
            raise ValueError("Pending/unreviewed/ungenerated poses cannot be published")
        intent = write_document(
            layout,
            layout.path(
                "campaigns", "pose-publications", request.sha256, "intent.json"
            ),
            request,
        )
        expected = PoseSnapshot(
            request=intent,
            ball_snapshot=request.ball_snapshot,
            clips=tuple(
                PublishedPose(
                    clip_id=c.canonical_clip_id or c.clip_id,
                    store_clip_id=c.store_clip_id,
                    result=results[c.canonical_clip_id or c.clip_id],
                )
                for c in ball.clips
            ),
        )
        validate_pose_snapshot(layout, expected)
        relative = f"ball_detection/{product.dataset_version}/player_pose"

        def build(_staging: Path, _destination: Path) -> PoseSnapshot:
            return expected

        def validate(snapshot: PoseSnapshot, _staging: Path | None) -> None:
            if snapshot != expected:
                raise ValueError("Pose snapshot differs from its publication intent")
            validate_pose_snapshot(layout, snapshot)

        snapshot_ref, snapshot = publish_directory(
            layout, relative, intent, kind=PoseSnapshot, build=build, validate=validate
        )
        receipt = PublicationReceipt(
            task="player",
            product="pose",
            request_id=request.sha256,
            base_snapshot=product.base_snapshot,
            snapshot=snapshot_ref,
            adoptions=tuple(adoptions),
            added_clips=tuple(added),
            reused_clips=tuple(c.clip_id for c in base.clips) if base else (),
        )
        reference = write_document(
            layout,
            layout.path(
                "campaigns", "pose-publications", request.sha256, "receipt.json"
            ),
            receipt,
        )
        StorageCatalog(layout).protect(
            protection(
                layout,
                identity="publication-v1-" + reference.digest,
                kind="publication",
                reason="approved poses, ID decisions and matching ball images",
                roots=(reference, intent),
                documents=(
                    receipt,
                    snapshot,
                    ball,
                    *documents,
                    *((base,) if base else ()),
                ),
            )
        )
        return reference


class PublishedPoseStore:
    """Receipt-bound reader. It never treats a pending result as an empty pose."""

    def __init__(self, layout: AnnotationLayout, receipt_ref: ArtifactRef) -> None:
        self.layout = layout
        self.receipt_ref = receipt_ref
        self.receipt = read_document(layout, receipt_ref, PublicationReceipt)
        if (self.receipt.task, self.receipt.product) != ("player", "pose"):
            raise ValueError("Pose reader requires a player/pose publication receipt")
        self.snapshot = read_document(layout, self.receipt.snapshot, PoseSnapshot)
        request = read_document(layout, self.snapshot.request, PosePublicationRequest)
        if (
            self.receipt.request_id != request.sha256
            or request.ball_snapshot != self.snapshot.ball_snapshot
        ):
            raise ValueError("Pose publication receipt and snapshot differ")
        validate_pose_snapshot(layout, self.snapshot)
        plan = load_plan(verify_reference(layout, request.plan))
        task = next(t for t in plan.tasks if t.task == "player")
        product = next(p for p in task.products if p.product == "pose")
        if (
            plan.phase != "production"
            or plan.annotation_root != str(layout.root)
            or (product.dataset_task, product.component)
            != ("ball_detection", "player_pose")
            or self.receipt.snapshot.path
            != f"ball_detection/{product.dataset_version}/player_pose/snapshot.json"
        ):
            raise ValueError(
                "Pose publication receipt differs from its planned location"
            )
        base = (
            read_document(layout, product.base_snapshot, PoseSnapshot)
            if product.base_snapshot
            else None
        )
        if base is not None:
            validate_pose_snapshot(layout, base)
        results = {c.clip_id: c.result for c in base.clips} if base else {}
        added, adoptions = [], []
        for reference in request.results:
            result, _ = validate_result(layout, reference)
            inputs, _, _ = load_final(layout, result.final_review)
            if (
                result.identity.clip_id in results
                or inputs.context.plan != request.plan
                or result.policy != product.generation_policy
            ):
                raise ValueError("Pose publication receipt has a different result/plan")
            results[result.identity.clip_id] = reference
            added.append(result.identity.clip_id)
            adoptions.append(inputs.binding.bbox_review)
        if (
            self.receipt.base_snapshot,
            self.receipt.added_clips,
            self.receipt.reused_clips,
            self.receipt.adoptions,
        ) != (
            product.base_snapshot,
            tuple(added),
            tuple(c.clip_id for c in base.clips) if base else (),
            tuple(adoptions),
        ) or results != {c.clip_id: c.result for c in self.snapshot.clips}:
            raise ValueError("Pose publication receipt membership differs from reader")
        self.clips = {c.clip_id: c for c in self.snapshot.clips}

    def read_clip(self, clip_id: str) -> dict | None:
        verify_reference(self.layout, self.receipt_ref)
        verify_reference(self.layout, self.receipt.snapshot)
        validate_pose_snapshot(self.layout, self.snapshot)
        _, arrays = validate_result(self.layout, self.clips[clip_id].result)
        return arrays
