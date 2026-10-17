"""Publish final bbox teachers, including confirmed empty and ignored frames."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

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
from src.tasks.base.annotation.publication import (
    final_reference,
    link_artifact,
    publish_directory,
)
from src.tasks.base.annotation.storage.catalog import StorageCatalog
from src.tasks.base.annotation.storage.references import protection
from src.tasks.player_detection.annotation.pose.corrected import (
    FinalReview,
    ReviewedInput,
    load_final,
)
from src.tasks.player_detection.annotation.review.contracts import PlayerLabels
from src.tasks.player_detection.data.snapshots.contracts import (
    ExistingPlayerSnapshot,
    SnapshotClip,
    SourceBinding,
)
from src.tasks.player_detection.data.snapshots.store import (
    clip_semantic_digest,
    load_snapshot,
    validate_snapshot,
)
from src.tasks.player_detection.data.store import (
    FRAME_FIELDS,
    INSTANCE_FIELDS,
    SCHEMA_VERSION,
    PlayerFrameStore,
    shard_name,
)
from src.tasks.player_detection.data.supervision import CODES, COLUMN, LEGACY, SCHEMA
from src.tasks.player_detection.generate_dataset.builder import ClipTables, merge_tables


class PublishRequest(Document):
    schema_version: Literal["player.bbox-publication.v1"] = "player.bbox-publication.v1"
    plan: ArtifactRef
    product: Literal["boxes"] = "boxes"
    reviews: tuple[ArtifactRef, ...]


class PlayerPublicationPolicy(Document):
    image_storage: Literal["reuse_ball_jpeg"]
    split: Literal["inherit_ball_source"]


def _tables(
    inputs: ReviewedInput, final: FinalReview, labels: PlayerLabels
) -> ClipTables:
    track_ids = tuple(
        sorted(
            {
                final.player_ids.get(o.observation_id, o.observation_id)
                for frame in labels.frames
                for o in frame.decision.observations
                if o.classification in ("player", "unresolved")
            }
        )
    )
    lookup = {name: index for index, name in enumerate(track_ids)}
    frames: dict[str, list[Any]] = {name: [] for name in FRAME_FIELDS}
    instances: dict[str, list[Any]] = {
        name: [] for name in (*INSTANCE_FIELDS, "bbox_xyxy")
    }
    for frame, mask in zip(labels.frames, final.teachers, strict=True):
        index = frame.frame.frame_index
        ball_row = inputs.store.row_of(inputs.clip, index)
        items = [
            o
            for o in frame.decision.observations
            if o.classification in ("player", "unresolved")
        ]
        for name, value in dict(
            clip=0,
            frame_index=index,
            source_frame_index=frame.frame.source_frame_index,
            clip_pts=frame.frame.clip_pts,
            is_target=frame.target,
            reviewed=frame.decision.reviewed,
            offset=inputs.store.frames["offset"][ball_row],
            length=inputs.store.frames["length"][ball_row],
            inst_start=0,
            inst_count=len(items),
            confirmed_empty=mask.supervision == "negative",
        ).items():
            frames[name].append(value)
        for item in items:
            instances["track_index"].append(
                lookup[final.player_ids.get(item.observation_id, item.observation_id)]
            )
            instances["bbox_source"].append(
                {"observed": 0, "estimated": 1, "unresolved": 2}[item.box_state]
            )
            instances["occluded"].append(False)
            instances["truncated"].append(False)
            instances["bbox_xyxy"].append(
                [item.box.x1, item.box.y1, item.box.x2, item.box.y2]
                if item.box
                else [np.nan] * 4
            )
    return ClipTables(
        {k: np.asarray(v, FRAME_FIELDS[k]) for k, v in frames.items()},
        {
            k: np.asarray(v, INSTANCE_FIELDS[k])
            for k, v in instances.items()
            if k != "bbox_xyxy"
        }
        | {"bbox_xyxy": np.asarray(instances["bbox_xyxy"], np.float32).reshape(-1, 4)},
        track_ids,
    )


def _metadata(
    inputs: ReviewedInput, final: FinalReview, table: ClipTables, index: int
) -> dict[str, Any]:
    manifest = inputs.manifest
    return dict(
        index=index,
        clip_id=manifest.clip_id,
        source_id=manifest.source.canonical_id,
        split=inputs.clip.split,
        width=manifest.width,
        height=manifest.height,
        frame_count=len(manifest.frames),
        time_base=manifest.time_base,
        nominal_fps=manifest.nominal_fps,
        track_ids=list(table.track_ids),
        annotation_status="partial"
        if any(r.target and r.supervision == "ignore" for r in final.teachers)
        else "completed",
        annotation_sha256=final.labels.digest,
        video_sha256=manifest.media.digest,
    )


def _validate_added(
    store: PlayerFrameStore,
    snapshot: ExistingPlayerSnapshot,
    additions: list[tuple[ReviewedInput, FinalReview, PlayerLabels]],
    tables: list[ClipTables],
    old_count: int,
) -> None:
    if store.teacher_supervision is None:
        raise ValueError("Player publication is missing its teacher masks")
    for offset, ((inputs, final, _), table) in enumerate(
        zip(additions, tables, strict=True), old_count
    ):
        rows = store.clip_frames(store.clips[offset])
        if cast(list[dict[str, Any]], store.metadata["clips"])[offset] != _metadata(
            inputs, final, table, offset
        ):
            raise ValueError("Published bbox provenance differs")
        start = int(store.frames["inst_start"][rows[0]])
        stop = start + int(store.frames["inst_count"][rows].sum())
        for key, values in table.frames.items():
            if key not in ("clip", "inst_start") and not np.array_equal(
                store.frames[key][rows], values
            ):
                raise ValueError("Published bbox frames differ from final ID decisions")
        for key, values in table.instances.items():
            if not np.array_equal(
                store.instances[key][start:stop], values, equal_nan=True
            ):
                raise ValueError(
                    "Published bbox teachers differ from final ID decisions"
                )
        if (
            not np.array_equal(
                store.teacher_supervision[rows],
                [CODES[r.supervision] for r in final.teachers],
            )
            or snapshot.clips[offset].shard.digest != inputs.shard.digest
        ):
            raise ValueError("Published teacher mask/RGB differs")


class PlayerPublisher:
    def __init__(self, layout: AnnotationLayout) -> None:
        if layout.review_namespace is not None:
            raise ValueError("Benchmark reviews cannot publish player datasets")
        self.layout = layout

    def publish(self, request: PublishRequest) -> ArtifactRef:
        layout = self.layout
        plan = load_plan(verify_reference(layout, request.plan))
        if plan.phase != "production" or plan.annotation_root != str(layout.root):
            raise ValueError("Player publication requires its fixed production plan")
        task = next((t for t in plan.tasks if t.task == "player"), None)
        product = (
            next((p for p in task.products if p.product == "boxes"), None)
            if task
            else None
        )
        if product is None or (product.dataset_task, product.component) != (
            "player_detection",
            "boxes",
        ):
            raise ValueError(
                "Player publication requires the planned player_detection/boxes product"
            )
        read_document(layout, product.generation_policy, PlayerPublicationPolicy)
        # Split assignments are inherited exactly from the corresponding ball snapshot.
        if not request.reviews or len(set(request.reviews)) != len(request.reviews):
            raise ValueError("Player publication needs unique final reviews")
        additions = sorted(
            (load_final(layout, r) for r in request.reviews),
            key=lambda a: a[0].manifest.clip_id,
        )
        if any(i.context.plan != request.plan for i, _, _ in additions):
            raise ValueError("Final bbox review belongs to another campaign plan")
        base, old = (
            load_snapshot(layout, product.base_snapshot)
            if product.base_snapshot
            else (None, None)
        )
        names = (
            {c.clip_id for c in base.clips} | {c.store_clip_id for c in base.clips}
            if base
            else set()
        )
        added_ids = [i.manifest.clip_id for i, _, _ in additions]
        if len(set(added_ids)) != len(added_ids) or names.intersection(added_ids):
            raise ValueError(
                "Player publication cannot duplicate or modify an existing clip"
            )
        splits = {c.source_id: c.split for c in base.clips} if base else {}
        for inputs, _, _ in additions:
            source = inputs.manifest.source.canonical_id
            if splits.setdefault(source, inputs.clip.split) != inputs.clip.split:
                raise ValueError(
                    "Player source split differs from the existing snapshot"
                )
        relative = f"player_detection/{product.dataset_version}" + (
            f"/{product.component}" if product.component else ""
        )
        intent = write_document(
            layout,
            layout.path(
                "campaigns", "player-publications", request.sha256, "intent.json"
            ),
            request,
        )
        tables = [_tables(*addition) for addition in additions]
        old_count = len(old.clips) if old else 0

        def build(staging: Path, destination: Path) -> ExistingPlayerSnapshot:
            (staging / "shards").mkdir()
            all_tables: list[ClipTables] = []
            metadata: list[dict[str, Any]] = []
            masks = []
            if old is not None:
                assert base is not None
                for clip, identity in zip(old.clips, base.clips, strict=True):
                    rows = old.clip_frames(clip)
                    start = int(old.frames["inst_start"][rows[0]])
                    stop = start + int(old.frames["inst_count"][rows].sum())
                    all_tables.append(
                        ClipTables(
                            {k: v[rows] for k, v in old.frames.items()},
                            {k: v[start:stop] for k, v in old.instances.items()},
                            clip.track_ids,
                        )
                    )
                    link_artifact(
                        layout,
                        identity.shard,
                        staging / "shards" / shard_name(clip.index),
                    )
                metadata.extend(cast(list[dict[str, Any]], old.metadata["clips"]))
                masks.append(
                    old.teacher_supervision
                    if old.teacher_supervision is not None
                    else np.full(len(old), LEGACY, np.uint8)
                )
            for offset, ((inputs, final, _), table) in enumerate(
                zip(additions, tables, strict=True), old_count
            ):
                all_tables.append(table)
                metadata.append(_metadata(inputs, final, table, offset))
                link_artifact(
                    layout, inputs.shard, staging / "shards" / shard_name(offset)
                )
                masks.append(
                    np.asarray([CODES[r.supervision] for r in final.teachers], np.uint8)
                )
            columns = merge_tables(all_tables)
            columns[COLUMN] = np.concatenate(masks)
            with (staging / "index.npz").open("xb") as output:
                np.savez(output, **columns)
                output.flush()
                os.fsync(output.fileno())
            with (staging / "metadata.json").open("x") as output:
                json.dump(
                    dict(
                        schema_version=SCHEMA_VERSION,
                        teacher_schema=SCHEMA,
                        version=product.dataset_version,
                        clips=metadata,
                        publication_request=intent.model_dump(mode="json"),
                        final_reviews=[
                            r.model_dump(mode="json") for r in request.reviews
                        ],
                        base_snapshot=product.base_snapshot.model_dump(mode="json")
                        if product.base_snapshot
                        else None,
                    ),
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            store = PlayerFrameStore(staging)
            identities = []
            for clip in store.clips:
                shard = final_reference(
                    layout,
                    staging / "shards" / shard_name(clip.index),
                    staging,
                    destination,
                )
                if base and clip.index < old_count:
                    identities.append(
                        base.clips[clip.index].model_copy(update={"shard": shard})
                    )
                else:
                    identities.append(
                        SnapshotClip(
                            clip_id=clip.clip_id,
                            canonical_clip_id=clip.clip_id,
                            store_clip_id=clip.clip_id,
                            source_id=clip.source_id,
                            split=clip.split,
                            semantic_digest=clip_semantic_digest(store, clip),
                            shard=shard,
                        )
                    )
            mapping = (
                {b.store_source_id: b.source_id for b in base.source_bindings}
                if base
                else {}
            )
            mapping.update(
                {
                    i.manifest.source.canonical_id: i.manifest.source.canonical_id
                    for i, _, _ in additions
                }
            )
            return ExistingPlayerSnapshot(
                index=final_reference(
                    layout, staging / "index.npz", staging, destination
                ),
                metadata=final_reference(
                    layout, staging / "metadata.json", staging, destination
                ),
                source_bindings=tuple(
                    SourceBinding(store_source_id=k, source_id=v)
                    for k, v in sorted(mapping.items())
                ),
                clips=tuple(identities),
                request=intent,
            )

        def validate(snapshot: ExistingPlayerSnapshot, staging: Path | None) -> None:
            if (
                snapshot.request != intent
                or snapshot.index.path != relative + "/index.npz"
            ):
                raise ValueError("Player snapshot belongs to another publication")
            store = validate_snapshot(layout, snapshot, staging=staging)
            if (
                len(store.clips) != old_count + len(additions)
                or store.teacher_supervision is None
            ):
                raise ValueError("Player publication membership/mask differs")
            if old is not None:
                assert base is not None
                if (
                    cast(list[dict[str, Any]], store.metadata["clips"])[:old_count]
                    != old.metadata["clips"]
                ):
                    raise ValueError("Existing player metadata changed")
                for previous, current in zip(
                    base.clips, snapshot.clips[:old_count], strict=True
                ):
                    if (
                        previous.model_dump(exclude={"shard"})
                        != current.model_dump(exclude={"shard"})
                        or previous.shard.digest != current.shard.digest
                    ):
                        raise ValueError("Existing player snapshot identity changed")
            _validate_added(store, snapshot, additions, tables, old_count)

        snapshot_ref, snapshot = publish_directory(
            layout,
            relative,
            intent,
            kind=ExistingPlayerSnapshot,
            build=build,
            validate=validate,
        )
        receipt = PublicationReceipt(
            task="player",
            product="boxes",
            request_id=request.sha256,
            base_snapshot=product.base_snapshot,
            snapshot=snapshot_ref,
            adoptions=tuple(i.binding.bbox_review for i, _, _ in additions),
            added_clips=tuple(added_ids),
            reused_clips=tuple(c.clip_id for c in base.clips) if base else (),
        )
        reference = write_document(
            layout,
            layout.path(
                "campaigns", "player-publications", request.sha256, "receipt.json"
            ),
            receipt,
        )
        StorageCatalog(layout).protect(
            protection(
                layout,
                identity="publication-v1-" + reference.digest,
                kind="publication",
                reason="final player bbox snapshot and immutable review inputs",
                roots=(reference, intent),
                documents=(
                    receipt,
                    request,
                    snapshot,
                    *((base,) if base else ()),
                    *(f for _, f, _ in additions),
                    *(i.input for i, _, _ in additions),
                ),
            )
        )
        return reference


def load_publication(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[PublicationReceipt, ExistingPlayerSnapshot, PlayerFrameStore]:
    """Read a receipt together with its exact base, final reviews and membership."""
    receipt = read_document(layout, reference, PublicationReceipt)
    if (receipt.task, receipt.product) != ("player", "boxes"):
        raise ValueError("Player bbox reader requires a player/boxes receipt")
    snapshot, store = load_snapshot(layout, receipt.snapshot)
    if snapshot.request is None:
        raise ValueError("Player publication is missing its frozen request")
    request = read_document(layout, snapshot.request, PublishRequest)
    plan = load_plan(verify_reference(layout, request.plan))
    task = next(t for t in plan.tasks if t.task == "player")
    product = next(p for p in task.products if p.product == "boxes")
    read_document(layout, product.generation_policy, PlayerPublicationPolicy)
    relative = f"player_detection/{product.dataset_version}/boxes"
    if (
        plan.phase != "production"
        or plan.annotation_root != str(layout.root)
        or (product.dataset_task, product.component) != ("player_detection", "boxes")
        or receipt.snapshot.path != relative + "/snapshot.json"
        or snapshot.index.path != relative + "/index.npz"
    ):
        raise ValueError("Player publication receipt differs from its planned location")
    base, _ = (
        load_snapshot(layout, product.base_snapshot)
        if product.base_snapshot
        else (None, None)
    )
    final = sorted(
        (load_final(layout, ref) for ref in request.reviews),
        key=lambda v: v[0].manifest.clip_id,
    )
    added = tuple(inputs.manifest.clip_id for inputs, _, _ in final)
    reused = tuple(c.clip_id for c in base.clips) if base else ()
    if (
        receipt.request_id,
        receipt.base_snapshot,
        receipt.added_clips,
        receipt.reused_clips,
        receipt.adoptions,
    ) != (
        request.sha256,
        product.base_snapshot,
        added,
        reused,
        tuple(inputs.binding.bbox_review for inputs, _, _ in final),
    ) or tuple(c.clip_id for c in snapshot.clips) != reused + added:
        raise ValueError(
            "Player publication receipt differs from the frozen reviews/snapshot"
        )
    if base and any(
        old.model_dump(exclude={"shard"}) != new.model_dump(exclude={"shard"})
        or old.shard.digest != new.shard.digest
        for old, new in zip(base.clips, snapshot.clips[: len(base.clips)], strict=True)
    ):
        raise ValueError("Player publication changed an existing snapshot")
    for at, (inputs, review, _) in enumerate(final, len(reused)):
        if (
            inputs.context.plan != request.plan
            or store.clips[at].annotation_sha256 != review.labels.digest
        ):
            raise ValueError("Player publication changed the final bbox provenance")
    _validate_added(
        store, snapshot, final, [_tables(*row) for row in final], len(reused)
    )
    return receipt, snapshot, store
