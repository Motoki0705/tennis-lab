"""Verified bbox revisions, lossless tracking input and final identity decisions.

The tracker proposes associations. Unassigned real observations remain explicit
singleton review units; they are never silently dropped or interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
    Document,
    content_digest,
)
from src.tasks.base.annotation.editing import ReviewContext, Revision, RevisionStore
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage.catalog import StorageCatalog
from src.tasks.base.annotation.storage.references import protection
from src.tasks.person_tracking.contracts import DetectionFeatures, ScoreOrigin
from src.tasks.person_tracking.features import FeatureExtractor
from src.tasks.person_tracking.sequence import DatasetTrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tasks.player_detection.annotation.review.adapter import PlayerReviewAdapter
from src.tasks.player_detection.annotation.review.contracts import (
    PlayerLabels,
    PlayerObservation,
    TeacherRow,
)
from src.tasks.player_detection.annotation.review.storage import retained_references
from src.tasks.player_detection.annotation.review.validation import teacher_rows

from .reviews import validate_and_remap
from .storage import write_npz


class BboxInput(Document):
    schema_version: Literal["player.corrected-bbox-input.v1"] = (
        "player.corrected-bbox-input.v1"
    )
    adoption: ArtifactRef
    ball_snapshot: ArtifactRef


class TrackingModels(Document):
    clip_reid: ArtifactRef
    aflink: ArtifactRef


class TrackingReceipt(Document):
    schema_version: Literal["player.corrected-tracking.v1"] = (
        "player.corrected-tracking.v1"
    )
    input: ArtifactRef
    models: TrackingModels
    tracks: ArtifactRef
    profile: dict[str, Any]
    observations: int
    unassigned_rows: tuple[int, ...]


class IdentityReview(Document):
    schema_version: Literal["player.identity-review.v1"] = "player.identity-review.v1"
    tracking: ArtifactRef
    decision: ArtifactRef
    # The ID reviewer records the images actually used for the decision.
    images: tuple[ArtifactRef, ...]
    required_sheets: tuple[str, ...]


class FinalReview(Document):
    schema_version: Literal["player.final-bbox-review.v1"] = (
        "player.final-bbox-review.v1"
    )
    input: ArtifactRef
    identity_review: ArtifactRef
    labels: ArtifactRef
    teachers: tuple[TeacherRow, ...]
    player_ids: dict[str, str]


@dataclass(frozen=True)
class ReviewedInput:
    reference: ArtifactRef
    input: BboxInput
    adoption: AdoptionReceipt
    context: ReviewContext
    manifest: ClipManifest
    labels: PlayerLabels
    store: BallFrameStore
    clip: ClipRecord
    shard: ArtifactRef


def load_input(layout: AnnotationLayout, reference: ArtifactRef) -> ReviewedInput:
    value = read_document(layout, reference, BboxInput)
    adoption = read_document(layout, value.adoption, AdoptionReceipt)
    if adoption.task != "player":
        raise ValueError("Corrected tracking requires a player bbox adoption")
    revision = read_document(layout, adoption.revision, Revision)
    context = read_document(layout, revision.context, ReviewContext)
    adapter = PlayerReviewAdapter()
    head, sealed = RevisionStore(layout).current(revision.context, adapter)
    if head != adoption.revision or sealed not in (None, value.adoption):
        raise ValueError("Bbox revision changed after tracking input was fixed")
    manifest = read_document(layout, context.manifest, ClipManifest)
    adapter.validate_adoption(layout, context, manifest, head, adoption)
    labels = read_document(layout, revision.payload, PlayerLabels)
    snapshot, store = load_snapshot(layout, value.ball_snapshot)
    candidates = [
        c
        for c in snapshot.clips
        if c.canonical_clip_id == manifest.clip_id or c.clip_id == manifest.clip_id
    ]
    if len(candidates) != 1:
        raise ValueError("Ball snapshot must identify the exact bbox clip")
    identity = candidates[0]
    clip = store.clip_by_id(identity.store_clip_id)
    rows = store.clip_rows(clip)
    if (
        identity.source_id,
        clip.media_sha256,
        clip.width,
        clip.height,
        clip.frame_count,
        Fraction(clip.time_base),
    ) != (
        manifest.source.canonical_id,
        manifest.media.digest,
        manifest.width,
        manifest.height,
        len(manifest.frames),
        Fraction(manifest.time_base),
    ):
        raise ValueError("Bbox media/geometry identity differs from ball snapshot")
    if clip.clip_id in store.video_frames:
        raise ValueError(
            "Player publication requires the planned native JPEG ball snapshot"
        )
    if not np.array_equal(
        store.frames["frame_index"][rows], [f.frame_index for f in manifest.frames]
    ) or not np.array_equal(
        store.frames["pts"][rows], [f.clip_pts for f in manifest.frames]
    ):
        raise ValueError("Bbox frame/PTS identity differs from ball snapshot")
    return ReviewedInput(
        reference,
        value,
        adoption,
        context,
        manifest,
        labels,
        store,
        clip,
        identity.shard,
    )


def pin_input(layout: AnnotationLayout, value: BboxInput) -> ArtifactRef:
    reference = write_document(
        layout,
        layout.path("annotations", "player-pose", "inputs", value.sha256 + ".json"),
        value,
    )
    inputs = load_input(layout, reference)
    StorageCatalog(layout).protect(
        protection(
            layout,
            identity="player-tracking-input-" + reference.digest,
            kind="active",
            reason="bbox adoption retained until final identity review",
            roots=(reference,),
            documents=(value, inputs.adoption, inputs.context, inputs.manifest),
            dependencies=retained_references(
                layout, inputs.context, inputs.adoption.revision, inputs.adoption
            ),
        )
    )
    return reference


def real_observations(labels: PlayerLabels) -> list[tuple[int, PlayerObservation]]:
    return [
        (frame.frame.frame_index, item)
        for frame in labels.frames
        for item in frame.decision.observations
        if item.classification == "player" and item.box_state == "observed"
    ]


def _raw_tracks(
    inputs: ReviewedInput, features: list[DetectionFeatures], aflink: AFLink
) -> tuple[dict[str, Any], tuple[int, ...]]:
    tracked = track_sequence(
        features,
        fps=float(Fraction(inputs.manifest.nominal_fps)),
        config=DatasetTrackingConfig(),
        aflink=aflink,
    )
    observations = real_observations(inputs.labels)
    rows = tracked.evidence.detection_rows.copy()
    ids = tracked.track_ids.copy()
    boxes = tracked.boxes.copy()
    seen = rows[rows >= 0]
    if len(set(seen.tolist())) != len(seen) or (seen >= len(observations)).any():
        raise ValueError("Tracker changed or duplicated an observation row")
    missing = tuple(sorted(set(range(len(observations))) - set(seen.tolist())))
    next_id = int(ids.max()) + 1 if len(ids) else 1
    for row in missing:
        frame, _ = observations[row]
        extra: np.ndarray = np.full((1, len(inputs.labels.frames)), -1, np.int64)
        extra[0, frame] = row
        rows = np.concatenate((rows, extra))
        boxes = np.concatenate(
            (boxes, np.zeros((1, len(inputs.labels.frames), 4), np.float32))
        )
        ids = np.append(ids, next_id).astype(np.int64)
        next_id += 1
    observation_ids = np.full(rows.shape, "", dtype="U64")
    origins = np.zeros(rows.shape, np.uint8)
    scores = np.zeros(rows.shape, np.float32)
    bbox_hashes = np.full(rows.shape, "", dtype="U64")
    for track, frame in zip(*np.nonzero(rows >= 0), strict=True):
        source_frame, item = observations[int(rows[track, frame])]
        if source_frame != frame:
            raise ValueError("Tracker moved an observation to another native frame")
        assert item.box is not None
        # The exact reviewed box is authoritative, never a Kalman estimate.
        boxes[track, frame] = [item.box.x1, item.box.y1, item.box.x2, item.box.y2]
        observation_ids[track, frame] = item.observation_id
        bbox_hashes[track, frame] = item.box.sha256
        scores[track, frame] = item.confidence
        origins[track, frame] = (
            ScoreOrigin.MODEL if item.score_origin == "model" else ScoreOrigin.MISSING
        )
    boxes[rows < 0] = 0
    return dict(
        frame_index=np.arange(len(inputs.labels.frames), dtype=np.int64),
        pts=np.asarray([f.frame.clip_pts for f in inputs.labels.frames], np.int64),
        source_pts=np.asarray(
            [f.frame.source_pts for f in inputs.labels.frames], np.int64
        ),
        track_ids=ids,
        boxes=boxes,
        detection_rows=rows,
        observation_ids=observation_ids,
        bbox_hashes=bbox_hashes,
        scores=scores,
        score_origin=origins,
    ), missing


def track_corrected(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    *,
    models: TrackingModels,
    extractor: FeatureExtractor,
    aflink: AFLink,
) -> ArtifactRef:
    """No detector is accepted or called. Models are supplied by the queue stage."""
    inputs = load_input(layout, reference)
    if extractor.pose is not None:
        raise ValueError("Corrected bbox tracking must not run a pose model")
    for weight in (models.clip_reid, models.aflink):
        verify_reference(layout, weight)
    identity = content_digest(
        [
            reference.model_dump(mode="json"),
            models.model_dump(mode="json"),
            DatasetTrackingConfig().identity(),
        ]
    )
    directory = layout.path(
        "annotations", "player-pose", inputs.manifest.clip_id, identity
    )
    receipt_path = directory / "tracking.json"
    with artifact_lock(layout, "corrected-tracking-" + identity):
        if receipt_path.exists():
            result = annotation_reference(layout, receipt_path)
            load_tracking(layout, result)
            return result
        features = []
        row = 0
        for frame in inputs.labels.frames:
            observations = [
                o
                for o in frame.decision.observations
                if o.classification == "player" and o.box_state == "observed"
            ]
            boxes = np.asarray(
                [
                    [o.box.x1, o.box.y1, o.box.x2, o.box.y2]
                    for o in observations
                    if o.box is not None
                ],
                np.float32,
            ).reshape(-1, 4)
            features.append(
                extractor.extract(
                    frame.frame.frame_index,
                    inputs.store.read_bgr(
                        inputs.store.row_of(inputs.clip, frame.frame.frame_index)
                    ),
                    np.arange(row, row + len(observations), dtype=np.int64),
                    boxes,
                    np.asarray([o.confidence for o in observations], np.float32),
                    score_origin=np.asarray(
                        [
                            ScoreOrigin.MODEL
                            if o.score_origin == "model"
                            else ScoreOrigin.MISSING
                            for o in observations
                        ],
                        np.uint8,
                    ),
                )
            )
            row += len(observations)
        raw, missing = _raw_tracks(inputs, features, aflink)
        directory.mkdir(parents=True, exist_ok=True)
        write_npz(directory / "tracks.npz", **raw)
        receipt = TrackingReceipt(
            input=reference,
            models=models,
            tracks=annotation_reference(layout, directory / "tracks.npz"),
            profile=DatasetTrackingConfig().identity(),
            observations=row,
            unassigned_rows=missing,
        )
        return write_document(layout, receipt_path, receipt)


def load_tracking(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ReviewedInput, TrackingReceipt, dict[str, Any]]:
    receipt = read_document(layout, reference, TrackingReceipt)
    inputs = load_input(layout, receipt.input)
    if receipt.profile != DatasetTrackingConfig().identity():
        raise ValueError("Corrected tracking profile changed")
    for weight in (receipt.models.clip_reid, receipt.models.aflink):
        verify_reference(layout, weight)
    with np.load(
        verify_reference(layout, receipt.tracks), allow_pickle=False
    ) as archive:
        raw = dict(archive)
    observations = real_observations(inputs.labels)
    rows = raw["detection_rows"]
    shape = (len(raw["track_ids"]), len(inputs.labels.frames))
    if (
        rows.dtype != np.int64
        or rows.shape != shape
        or raw["boxes"].shape != (*shape, 4)
        or len(set(raw["track_ids"].tolist())) != shape[0]
    ):
        raise ValueError("Tracking axes/track identities differ")
    if (
        any(
            raw[name].shape != shape
            for name in ("observation_ids", "bbox_hashes", "scores", "score_origin")
        )
        or raw["boxes"][rows < 0].any()
    ):
        raise ValueError("Tracking provenance axes/missing mask differ")
    if sorted(rows[rows >= 0].tolist()) != list(
        range(len(observations))
    ) or receipt.observations != len(observations):
        raise ValueError(
            "Tracking must retain every corrected observation exactly once"
        )
    if (
        not np.array_equal(raw["pts"], [f.frame.clip_pts for f in inputs.labels.frames])
        or not np.array_equal(
            raw["source_pts"], [f.frame.source_pts for f in inputs.labels.frames]
        )
        or not np.array_equal(raw["frame_index"], np.arange(len(inputs.labels.frames)))
    ):
        raise ValueError("Tracking frame/PTS changed")
    for track, frame in zip(*np.nonzero(rows >= 0), strict=True):
        expected_frame, item = observations[int(rows[track, frame])]
        assert item.box is not None
        box = np.asarray(
            [item.box.x1, item.box.y1, item.box.x2, item.box.y2], np.float32
        )
        if (
            expected_frame != frame
            or raw["observation_ids"][track, frame] != item.observation_id
            or raw["bbox_hashes"][track, frame] != item.box.sha256
            or not np.array_equal(raw["boxes"][track, frame], box)
        ):
            raise ValueError("Tracking changed corrected bbox/observation identity")
        if raw["score_origin"][track, frame] != (
            ScoreOrigin.MODEL if item.score_origin == "model" else ScoreOrigin.MISSING
        ) or raw["scores"][track, frame] != np.float32(item.confidence):
            raise ValueError("Tracking changed score provenance")
    return inputs, receipt, raw


def review_selection(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ReviewedInput, IdentityReview, dict[str, Any], dict[str, Any]]:
    review = read_document(layout, reference, IdentityReview)
    inputs, receipt, raw = load_tracking(layout, review.tracking)
    if (
        not review.images
        or tuple(p.path.rsplit("/", 1)[-1] for p in review.images)
        != review.required_sheets
    ):
        raise ValueError("ID review requires its exact image sheets")
    for image in review.images:
        verify_reference(layout, image)
    import json

    decision = json.loads(verify_reference(layout, review.decision).read_bytes())
    selected = validate_and_remap(
        raw,
        decision,
        clip_id=inputs.manifest.clip_id,
        raw_hash=receipt.tracks.digest,
        required_sheets=list(review.required_sheets),
    )
    return inputs, review, raw, selected


def final_labels(
    inputs: ReviewedInput, raw: dict[str, Any], decision: dict[str, Any]
) -> PlayerLabels:
    changes: dict[str, tuple[str, str | None]] = {}
    selected_ids: dict[tuple[int, str], str] = {}
    lookup = {int(value): at for at, value in enumerate(raw["track_ids"])}
    for segment in decision["segments"]:
        at = lookup[segment["raw_track_id"]]
        for frame in range(segment["start_frame"], segment["stop_frame"]):
            if raw["detection_rows"][at, frame] < 0:
                continue
            identity = str(raw["observation_ids"][at, frame])
            changes[identity] = segment["role"], segment["player_id"]
            if segment["role"] == "player":
                selected_ids[frame, segment["player_id"]] = identity
    frames = []
    for frame in inputs.labels.frames:
        items = []
        for item in frame.decision.observations:
            if item.observation_id in changes:
                role, player = changes[item.observation_id]
                if role != "player":
                    item = item.model_copy(
                        update=dict(
                            classification="unresolved" if role == "unknown" else role,
                            box_state="unresolved"
                            if role == "unknown"
                            else "not_applicable",
                            box=None,
                            origin="manual",
                            score_origin="missing",
                            confidence=0.0,
                            duplicate_of=selected_ids[
                                frame.frame.frame_index, str(player)
                            ]
                            if role == "duplicate"
                            else None,
                        )
                    )
            items.append(item)
        kept = {o.observation_id for o in items if o.classification == "player"}
        if any(
            o.classification == "duplicate" and o.duplicate_of not in kept
            for o in items
        ):
            raise ValueError(
                "ID review invalidates a prior duplicate; revise its bbox decision explicitly"
            )
        frames.append(
            frame.model_copy(
                update={
                    "decision": frame.decision.model_copy(
                        update={"observations": tuple(items)}
                    )
                }
            )
        )
    result = inputs.labels.model_copy(update={"frames": tuple(frames)})
    return PlayerLabels.model_validate_json(result.model_dump_json())


def finalize_review(layout: AnnotationLayout, reference: ArtifactRef) -> ArtifactRef:
    import json

    inputs, review, raw, selected = review_selection(layout, reference)
    decision = json.loads(verify_reference(layout, review.decision).read_bytes())
    if decision["status"] != "approved":
        raise ValueError("Final bbox publication requires completed identity review")
    labels = final_labels(inputs, raw, decision)
    directory = layout.path(
        "annotations",
        "player-pose",
        inputs.manifest.clip_id,
        "final",
        inputs.reference.digest,
    )
    observations = real_observations(inputs.labels)
    player_ids = {
        observations[int(selected["detection_rows"][frame, player])][
            1
        ].observation_id: str(selected["player_ids"][player])
        for frame, player in zip(*np.nonzero(selected["observed"]), strict=True)
    }
    with artifact_lock(layout, "final-player-review-" + inputs.reference.digest):
        labels_ref = write_document(layout, directory / "labels.json", labels)
        value = FinalReview(
            input=inputs.reference,
            identity_review=reference,
            labels=labels_ref,
            teachers=teacher_rows(labels),
            player_ids=player_ids,
        )
        RevisionStore(layout).seal(
            inputs.adoption.revision, inputs.input.adoption, PlayerReviewAdapter()
        )
        result = write_document(layout, directory / "receipt.json", value)
        tracking = read_document(layout, review.tracking, TrackingReceipt)
        StorageCatalog(layout).protect(
            protection(
                layout,
                identity="final-player-review-" + result.digest,
                kind="adoption",
                reason="final ID decisions, corrected tracking and reviewed bbox teachers",
                roots=(result,),
                documents=(value, review, tracking, inputs.input),
            )
        )
        return result


def load_final(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ReviewedInput, FinalReview, PlayerLabels]:
    import json

    value = read_document(layout, reference, FinalReview)
    inputs, review, raw, selected = review_selection(layout, value.identity_review)
    revision = read_document(layout, inputs.adoption.revision, Revision)
    if (
        RevisionStore(layout).current(revision.context, PlayerReviewAdapter())[1]
        != inputs.input.adoption
    ):
        raise ValueError("Final bbox review requires a sealed adoption")
    decision = json.loads(verify_reference(layout, review.decision).read_bytes())
    labels = final_labels(inputs, raw, decision)
    observations = real_observations(inputs.labels)
    identities = {
        observations[int(selected["detection_rows"][frame, player])][
            1
        ].observation_id: str(selected["player_ids"][player])
        for frame, player in zip(*np.nonzero(selected["observed"]), strict=True)
    }
    if (
        decision["status"] != "approved"
        or inputs.reference != value.input
        or value.teachers != teacher_rows(labels)
        or value.player_ids != identities
        or read_document(layout, value.labels, PlayerLabels) != labels
    ):
        raise ValueError("Final bbox receipt differs from its identity review")
    return inputs, value, labels
