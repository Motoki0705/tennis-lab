"""Publish one cumulative snapshot, recovering rename/receipt interruption gaps."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.tasks.ball_detection.data.store import BallFrameStore, Split, shard_name
from src.tasks.ball_detection.data.teacher_masks import EXPLICIT_MASK_SCHEMA, LEGACY
from src.tasks.ball_detection.data.windows.manifest import clip_semantic_digest
from src.tasks.ball_detection.generate_dataset.frame_store.builder import (
    EncodedClip,
    StoredSize,
    encode_clip,
    merge_tables,
)
from src.tasks.base.annotation.artifacts import (
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    PublicationReceipt,
    content_digest,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import ProductPlan, load_plan
from src.tasks.base.annotation.registry import Registry

from .adopted import AdoptedClip, load_adopted
from .contracts import BallPublicationPolicy, BallSnapshot, PublishRequest, SnapshotClip
from .snapshot import data_reference, load_snapshot, validate_snapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ball_publications (
    request_id TEXT PRIMARY KEY,
    destination TEXT NOT NULL UNIQUE,
    intent TEXT NOT NULL,
    receipt TEXT
);
"""


@dataclass(frozen=True)
class Inputs:
    product: ProductPlan
    policy: BallPublicationPolicy
    base: BallSnapshot | None
    store: BallFrameStore | None
    additions: tuple[AdoptedClip, ...]
    splits: dict[str, Split]


def _durable_document(path: Path, document: Document | dict[str, Any]) -> None:
    payload = (
        document.canonical_bytes()
        if isinstance(document, Document)
        else json.dumps(
            document, ensure_ascii=False, sort_keys=True, allow_nan=False
        ).encode()
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".write-{path.name}-", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(payload + b"\n")
            output.flush()
            os.fsync(output.fileno())
        # A document appears whole, and never replaces an existing document.
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_version(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _link_verified(
    layout: AnnotationLayout, reference: ArtifactRef, target: Path
) -> None:
    source = layout.artifact_path(reference)
    before = _file_version(source)
    verify_reference(layout, reference)
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise ValueError(
                "ball snapshot requires hardlinks on the same filesystem; copying RGB is not automatic"
            ) from error
        raise
    if before != _file_version(source) or before != _file_version(target):
        raise ValueError("base RGB shard changed while linking")


def _clip_record(added: AdoptedClip, index: int, split: Split) -> dict[str, Any]:
    spec = added.spec
    return dict(
        index=index,
        clip_id=spec.clip_id,
        source=spec.source,
        group_id=spec.group_id,
        camera_id=spec.camera_id,
        split=split,
        width=spec.width,
        height=spec.height,
        source_width=spec.width,
        source_height=spec.height,
        frame_count=spec.labels.frame_count,
        time_base=str(spec.time_base),
        fps=str(spec.fps),
        has_events=spec.has_events,
        track_ids=list(spec.labels.track_ids),
        annotation_path=str(spec.annotation_path),
        annotation_sha256=spec.annotation_sha256,
        media_sha256=spec.media_sha256,
        provenance=spec.provenance,
    )


class BallPublisher:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.registry = Registry(layout)
        self.registry.initialize("ball-publications-v1", _SCHEMA)

    def _inputs(self, request: PublishRequest) -> Inputs:
        plan = load_plan(verify_reference(self.layout, request.plan))
        if plan.annotation_root != str(self.layout.root) or not plan.allows_adoption:
            raise ValueError(
                "ball publication requires a production plan at the fixed root"
            )
        task = next((t for t in plan.tasks if t.task == "ball"), None)
        product = (
            next((p for p in task.products if p.product == request.product), None)
            if task
            else None
        )
        if product is None or (product.dataset_task, product.component) != (
            "ball_detection",
            "frames",
        ):
            raise ValueError(
                "ball publisher requires the planned ball_detection/frames product"
            )
        policy = read_document(
            self.layout, product.generation_policy, BallPublicationPolicy
        )
        base, store = (
            load_snapshot(self.layout, product.base_snapshot)
            if product.base_snapshot
            else (None, None)
        )
        additions = tuple(
            sorted(
                (
                    load_adopted(self.layout, ref, request.plan)
                    for ref in request.adoptions
                ),
                key=lambda a: a.spec.clip_id,
            )
        )
        existing_ids = {c.clip_id for c in base.clips} if base else set()
        existing_names = {c.store_clip_id for c in base.clips} if base else set()
        canonical_ids = {c.canonical_clip_id for c in base.clips} if base else set()
        added_ids = [a.spec.clip_id for a in additions]
        if len(set(added_ids)) != len(added_ids) or (
            set(added_ids) & (existing_ids | existing_names | canonical_ids)
        ):
            raise ValueError("publication cannot duplicate or change an existing clip")
        splits: dict[str, Split] = (
            {c.source_id: c.split for c in base.clips} if base else {}
        )
        for entry in policy.source_splits:
            if entry.source_id in splits and splits[entry.source_id] != entry.split:
                raise ValueError(
                    "publication policy cannot change an existing source split"
                )
            splits[entry.source_id] = entry.split
        if any(a.source_id not in splits for a in additions):
            raise ValueError("every new source requires an explicit fixed split")
        return Inputs(product, policy, base, store, additions, splits)

    def _build(
        self,
        inputs: Inputs,
        request_ref: ArtifactRef,
        directory: Path,
        destination: Path,
    ) -> BallSnapshot:
        (directory / "shards").mkdir()
        tables: list[EncodedClip] = []
        clips: list[dict[str, Any]] = []
        heatmaps, coordinates = [], []
        base = inputs.store
        if base is not None:
            assert inputs.base is not None
            clips = cast(list[dict[str, Any]], base.metadata["clips"]).copy()
            for clip, identity in zip(base.clips, inputs.base.clips, strict=True):
                rows = base.clip_rows(clip)
                first = int(base.frames["inst_start"][rows[0]])
                stop = first + int(base.frames["inst_count"][rows].sum())
                tables.append(
                    EncodedClip(
                        {name: value[rows] for name, value in base.frames.items()},
                        {
                            name: value[first:stop]
                            for name, value in base.instances.items()
                        },
                    )
                )
                _link_verified(
                    self.layout,
                    identity.shard,
                    directory / "shards" / shard_name(clip.index),
                )
            heatmaps.append(
                np.full(len(base), LEGACY, np.uint8)
                if base.teacher_masks is None
                else base.teacher_masks.heatmap
            )
            coordinates.append(
                np.full(len(base), LEGACY, np.uint8)
                if base.teacher_masks is None
                else base.teacher_masks.coordinate
            )
        for added in inputs.additions:
            spec, index = added.spec, len(tables)
            shard = directory / "shards" / shard_name(index)
            tables.append(
                encode_clip(
                    spec,
                    shard,
                    inputs.policy.jpeg_quality,
                    StoredSize(spec.width, spec.height),
                )
            )
            with shard.open("rb") as stored:
                os.fsync(stored.fileno())
            clips.append(_clip_record(added, index, inputs.splits[added.source_id]))
            heatmaps.append(added.heatmap)
            coordinates.append(added.coordinate)
        columns = merge_tables(tables)
        columns["teacher_heatmap"] = np.concatenate(heatmaps)
        columns["teacher_coordinate"] = np.concatenate(coordinates)
        with (directory / "index.npz").open("xb") as output:
            np.savez(output, **columns)
            output.flush()
            os.fsync(output.fileno())
        history = (
            cast(list[dict[str, Any]], base.metadata.get("append_history", [])).copy()
            if base
            else []
        )
        history.append(
            dict(
                request=request_ref.model_dump(mode="json"),
                base_snapshot=inputs.product.base_snapshot.model_dump(mode="json")
                if inputs.product.base_snapshot
                else None,
            )
        )
        _durable_document(
            directory / "metadata.json",
            dict(
                schema_version=EXPLICIT_MASK_SCHEMA,
                version=inputs.product.dataset_version,
                component="frames",
                jpeg_quality=inputs.policy.jpeg_quality,
                clips=clips,
                append_history=history,
                publication_request=request_ref.model_dump(mode="json"),
            ),
        )
        store = BallFrameStore(directory)

        def final_reference(path: Path) -> ArtifactRef:
            reference = data_reference(self.layout, path)
            final = destination / path.relative_to(directory)
            return ArtifactRef(
                role="data",
                path=final.relative_to(self.layout.repository / "data").as_posix(),
                digest=reference.digest,
                size_bytes=reference.size_bytes,
            )

        identities: list[SnapshotClip] = []
        for clip in store.clips:
            if inputs.base and clip.index < len(inputs.base.clips):
                previous = inputs.base.clips[clip.index]
                clip_id, source_id = previous.clip_id, previous.source_id
                canonical_clip_id = previous.canonical_clip_id
            else:
                clip_id, source_id = clip.clip_id, clip.group_id
                canonical_clip_id = clip.clip_id
            identities.append(
                SnapshotClip(
                    clip_id=clip_id,
                    canonical_clip_id=canonical_clip_id,
                    store_clip_id=clip.clip_id,
                    source_id=source_id,
                    split=clip.split,
                    semantic_digest=clip_semantic_digest(store, clip),
                    shard=final_reference(
                        directory / "shards" / shard_name(clip.index)
                    ),
                )
            )
        snapshot = BallSnapshot(
            index=final_reference(directory / "index.npz"),
            metadata=final_reference(directory / "metadata.json"),
            clips=tuple(identities),
            request=request_ref,
            legacy_groups=(),
        )
        _durable_document(directory / "snapshot.json", snapshot)
        sync_directory(directory / "shards")
        sync_directory(directory)
        return snapshot

    def _validate_result(
        self,
        inputs: Inputs,
        request_ref: ArtifactRef,
        snapshot: BallSnapshot,
        *,
        staging: Path | None = None,
    ) -> None:
        if snapshot.request != request_ref:
            raise ValueError("published snapshot belongs to another request")
        expected_index = (
            f"ball_detection/{inputs.product.dataset_version}/frames/index.npz"
        )
        if snapshot.index.path != expected_index:
            raise ValueError("snapshot files differ from the planned destination")
        store = validate_snapshot(self.layout, snapshot, staging=staging)
        expected_count = (len(inputs.base.clips) if inputs.base else 0) + len(
            inputs.additions
        )
        if len(store.clips) != expected_count or store.teacher_masks is None:
            raise ValueError(
                "published snapshot has the wrong membership or mask schema"
            )
        base = inputs.store
        old_count = len(base.clips) if base else 0
        if base is not None:
            assert inputs.base is not None
            if (
                cast(list[dict[str, Any]], store.metadata["clips"])[:old_count]
                != base.metadata["clips"]
            ):
                raise ValueError("publication changed existing clip metadata")
            for name, values in (*base.frames.items(), *base.instances.items()):
                actual = (
                    store.frames[name] if name in base.frames else store.instances[name]
                )
                if not np.array_equal(actual[: len(values)], values, equal_nan=True):
                    raise ValueError(f"publication changed existing column {name}")
            for name in ("heatmap", "coordinate"):
                expected = (
                    np.full(len(base), LEGACY, np.uint8)
                    if base.teacher_masks is None
                    else getattr(base.teacher_masks, name)
                )
                if not np.array_equal(
                    getattr(store.teacher_masks, name)[: len(base)], expected
                ):
                    raise ValueError("publication changed existing teacher masks")
            for previous, current in zip(
                inputs.base.clips, snapshot.clips[:old_count], strict=True
            ):
                if (
                    previous.model_dump(exclude={"shard"})
                    != current.model_dump(exclude={"shard"})
                    or previous.shard.digest != current.shard.digest
                ):
                    raise ValueError("publication changed existing snapshot identities")
        for offset, added in enumerate(inputs.additions, old_count):
            clip, spec = store.clips[offset], added.spec
            identity = snapshot.clips[offset]
            if (identity.clip_id, identity.source_id) != (
                spec.clip_id,
                added.source_id,
            ) or cast(list[dict[str, Any]], store.metadata["clips"])[
                offset
            ] != _clip_record(added, offset, inputs.splits[added.source_id]):
                raise ValueError("published clip differs from its adopted provenance")
            rows = store.clip_rows(clip)
            first = int(store.frames["inst_start"][rows[0]])
            stop = first + int(store.frames["inst_count"][rows].sum())
            for name, values in spec.labels.frames.items():
                if not np.array_equal(store.frames[name][rows], values, equal_nan=True):
                    raise ValueError("published frames differ from adopted labels")
            for name, values in dict(
                spec.labels.instances, xy=spec.labels.xy.astype(np.float32)
            ).items():
                if not np.array_equal(
                    store.instances[name][first:stop], values, equal_nan=True
                ):
                    raise ValueError("published instances differ from adopted labels")
            if not np.array_equal(
                store.teacher_masks.heatmap[rows], added.heatmap
            ) or not np.array_equal(
                store.teacher_masks.coordinate[rows], added.coordinate
            ):
                raise ValueError("published teachers differ from the adoption mask")

    def publish(self, request: PublishRequest) -> ArtifactRef:
        inputs = self._inputs(request)
        relative = f"ball_detection/{inputs.product.dataset_version}/frames"
        destination = self.layout.artifact_path(
            ArtifactRef(
                role="data",
                path=relative + "/snapshot.json",
                digest="0" * 64,
                size_bytes=0,
            )
        ).parent
        if inputs.store is not None and (
            destination.is_relative_to(inputs.store.directory)
            or inputs.store.directory.is_relative_to(destination)
        ):
            raise ValueError(
                "publication must use a new version outside the base store"
            )
        with artifact_lock(self.layout, "ball-publication-" + content_digest(relative)):
            intent = write_document(
                self.layout,
                self.layout.path(
                    "campaigns", "ball-publications", request.sha256, "intent.json"
                ),
                request,
            )
            with self.registry.transaction() as db:
                existing = db.execute(
                    "SELECT * FROM ball_publications WHERE destination=? OR request_id=?",
                    (relative, request.sha256),
                ).fetchall()
                if existing and (
                    len(existing) != 1
                    or existing[0]["request_id"] != request.sha256
                    or existing[0]["intent"] != intent.model_dump_json()
                ):
                    raise ValueError(
                        "publication destination is reserved by another request"
                    )
                if not existing:
                    db.execute(
                        "INSERT INTO ball_publications VALUES (?,?,?,NULL)",
                        (request.sha256, relative, intent.model_dump_json()),
                    )
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = destination.with_name(".frames.staging-" + request.sha256)
            if destination.exists():
                snapshot = BallSnapshot.model_validate_json(
                    (destination / "snapshot.json").read_bytes()
                )
            else:
                owner = staging / "publication-intent.json"
                if staging.exists():
                    self.layout.artifact_path(
                        ArtifactRef(
                            role="data",
                            path=owner.relative_to(
                                self.layout.repository / "data"
                            ).as_posix(),
                            digest="0" * 64,
                            size_bytes=0,
                        )
                    )
                    if not owner.exists():
                        # The directory/owner creation gap is recoverable only
                        # before any data files exist, under the reserved intent.
                        remnants = tuple(staging.iterdir())
                        if any(
                            not p.name.startswith(".write-publication-intent.json-")
                            or not p.is_file()
                            or p.is_symlink()
                            for p in remnants
                        ):
                            raise ValueError(
                                "staging directory has no publication owner"
                            )
                        for path in remnants:
                            path.unlink()
                        _durable_document(owner, intent)
                    if owner.read_bytes() != intent.canonical_bytes() + b"\n":
                        raise ValueError(
                            "staging directory is not owned by this publication"
                        )
                    if not (staging / "snapshot.json").exists():
                        # Only this intent's incomplete index/new JPEG staging is
                        # rebuilt; published/base artifacts are never removed.
                        for path in staging.iterdir():
                            if path == owner:
                                continue
                            if path.is_dir() and not path.is_symlink():
                                shutil.rmtree(path)
                            else:
                                path.unlink()
                        sync_directory(staging)
                if not staging.exists():
                    staging.mkdir()
                    _durable_document(owner, intent)
                if not (staging / "snapshot.json").exists():
                    snapshot = self._build(inputs, intent, staging, destination)
                else:
                    snapshot = BallSnapshot.model_validate_json(
                        (staging / "snapshot.json").read_bytes()
                    )
                self._validate_result(inputs, intent, snapshot, staging=staging)
                if destination.exists():
                    raise ValueError("publication destination appeared during staging")
                os.rename(staging, destination)
                sync_directory(destination.parent)
            self._validate_result(inputs, intent, snapshot)
            snapshot_ref = data_reference(self.layout, destination / "snapshot.json")
            receipt = PublicationReceipt(
                task="ball",
                product=request.product,
                request_id=request.sha256,
                base_snapshot=inputs.product.base_snapshot,
                snapshot=snapshot_ref,
                adoptions=request.adoptions,
                added_clips=tuple(a.spec.clip_id for a in inputs.additions),
                reused_clips=tuple(c.clip_id for c in inputs.base.clips)
                if inputs.base
                else (),
            )
            reference = write_document(
                self.layout,
                self.layout.path(
                    "campaigns", "ball-publications", request.sha256, "receipt.json"
                ),
                receipt,
            )
            with self.registry.transaction() as db:
                current = db.execute(
                    "SELECT receipt FROM ball_publications WHERE request_id=?",
                    (request.sha256,),
                ).fetchone()
                if (
                    current["receipt"] is not None
                    and current["receipt"] != reference.model_dump_json()
                ):
                    raise ValueError(
                        "registered publication receipt differs from completed data"
                    )
                db.execute(
                    "UPDATE ball_publications SET receipt=? WHERE request_id=?",
                    (reference.model_dump_json(), request.sha256),
                )
            result: ArtifactRef = reference
            return result
