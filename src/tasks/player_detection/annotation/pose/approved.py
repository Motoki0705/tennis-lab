"""Resumable pose chunks bound to exact reviewed observations and image bytes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import Field

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    content_digest,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.person_tracking.features import PoseModel

from .corrected import load_final, review_selection
from .dataset import validate_pose_arrays
from .selected_pose import infer_selected_frame
from .selection import presence_counts
from .storage import write_npz


class PosePolicy(Document):
    presence_threshold: float = Field(ge=0, le=1)
    chunk_frames: int = Field(gt=0)


class PoseIdentity(Document):
    clip_id: Digest
    media: Digest
    frames: Digest
    image: Digest
    bbox_revision: Digest
    final_labels: Digest
    id_review: Digest
    selection: Digest
    weights: ArtifactRef


class PoseResult(Document):
    schema_version: Literal["player.approved-pose.v1"] = "player.approved-pose.v1"
    final_review: ArtifactRef
    policy: ArtifactRef
    identity: PoseIdentity
    status: Literal["generated", "reused", "not_applicable"]
    reason: str | None
    poses: ArtifactRef | None
    reused_from: ArtifactRef | None


class PoseChunk(Document):
    identity: PoseIdentity
    start: int
    stop: int
    data: ArtifactRef


def _identity(
    layout: AnnotationLayout, final: ArtifactRef, weights: ArtifactRef
) -> tuple[PoseIdentity, Any, dict[str, Any]]:
    inputs, value, _ = load_final(layout, final)
    _, review, raw, selected = review_selection(layout, value.identity_review)
    for name in ("observation_ids", "bbox_hashes", "scores", "score_origin"):
        source = raw[name]
        dtype = source.dtype
        result = np.zeros(selected["observed"].shape, dtype=dtype)
        lookup = {
            int(row): (track, frame)
            for track, frame in zip(
                *np.nonzero(raw["detection_rows"] >= 0), strict=True
            )
            for row in [raw["detection_rows"][track, frame]]
        }
        for frame, player in zip(*np.nonzero(selected["observed"]), strict=True):
            track, source_frame = lookup[int(selected["detection_rows"][frame, player])]
            result[frame, player] = source[track, source_frame]
        selected[name] = result
    selected["source_pts"] = raw["source_pts"]
    verify_reference(layout, weights)
    identity = PoseIdentity(
        clip_id=inputs.manifest.clip_id,
        media=inputs.manifest.media.digest,
        frames=content_digest(
            [f.model_dump(mode="json") for f in inputs.manifest.frames]
        ),
        image=inputs.shard.digest,
        bbox_revision=inputs.adoption.revision.digest,
        final_labels=value.labels.digest,
        id_review=value.identity_review.digest,
        selection=content_digest({k: v.tolist() for k, v in selected.items()}),
        weights=weights,
    )
    return identity, inputs, selected


def validate_result(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[PoseResult, dict[str, Any] | None]:
    result = read_document(layout, reference, PoseResult)
    policy = read_document(layout, result.policy, PosePolicy)
    identity, inputs, selection = _identity(
        layout, result.final_review, result.identity.weights
    )
    if result.identity != identity:
        raise ValueError(
            "Pose identity changed: media/frame/PTS/image/bbox/review/weights"
        )
    presence = presence_counts(inputs.store, policy.presence_threshold)[
        inputs.clip.index
    ]
    excluded = not presence["selected"] or not selection["observed"].any()
    if result.status == "not_applicable":
        expected = (
            presence["reason"]
            if not presence["selected"]
            else "no_approved_real_players"
        )
        if (
            not excluded
            or result.reason != expected
            or result.poses is not None
            or result.reused_from is not None
        ):
            raise ValueError(
                "Pose exclusion must match the explicit policy and observations"
            )
        return result, None
    if excluded or result.poses is None or result.reason is not None:
        raise ValueError("Generated pose outcome disagrees with its generation policy")
    if (result.status == "reused") != (result.reused_from is not None):
        raise ValueError("Reused pose requires an exact prior result")
    if result.reused_from is not None:
        previous, _ = validate_result(layout, result.reused_from)
        if previous.identity != identity or previous.poses != result.poses:
            raise ValueError("Reused pose input identity differs")
    with np.load(verify_reference(layout, result.poses), allow_pickle=False) as archive:
        arrays = dict(archive)
    if set(arrays) != {*selection, "keypoints"} or any(
        not np.array_equal(arrays[k], v) for k, v in selection.items()
    ):
        raise ValueError("Pose artifact differs from its exact reviewed observations")
    if arrays["keypoints"].dtype != np.float32:
        raise ValueError("Pose output must retain float32 native keypoints")
    validate_pose_arrays(arrays, inputs.store, inputs.clip)
    return result, arrays


def generate_pose(
    layout: AnnotationLayout,
    final: ArtifactRef,
    *,
    policy_ref: ArtifactRef,
    weights: ArtifactRef,
    model_factory: Callable[[], PoseModel],
    reuse: ArtifactRef | None = None,
) -> ArtifactRef:
    """The caller owns CPU/GPU admission; model_factory is never called for exclusions/reuse."""
    policy = read_document(layout, policy_ref, PosePolicy)
    identity, inputs, selection = _identity(layout, final, weights)
    request = content_digest(
        [
            final.model_dump(mode="json"),
            policy_ref.model_dump(mode="json"),
            weights.model_dump(mode="json"),
            reuse.model_dump(mode="json") if reuse else None,
        ]
    )
    directory = layout.path(
        "annotations", "player-pose", inputs.manifest.clip_id, "poses", request
    )
    with artifact_lock(layout, "approved-pose-" + request):
        if (directory / "receipt.json").exists():
            ref = annotation_reference(layout, directory / "receipt.json")
            validate_result(layout, ref)
            return ref
        presence = presence_counts(inputs.store, policy.presence_threshold)[
            inputs.clip.index
        ]
        reason = (
            presence["reason"]
            if not presence["selected"]
            else "no_approved_real_players"
            if not selection["observed"].any()
            else None
        )
        pose_ref = None
        status: Literal["generated", "reused", "not_applicable"] = (
            "not_applicable" if reason else "generated"
        )
        if reuse is not None:
            previous, _ = validate_result(layout, reuse)
            if (
                reason is not None
                or previous.identity != identity
                or previous.poses is None
            ):
                raise ValueError(
                    "Approved pose reuse requires identical clip/media/frame/PTS/image/bbox/review/weights"
                )
            pose_ref, status = previous.poses, "reused"
        elif reason is None:
            output = np.zeros((*selection["observed"].shape, 17, 3), np.float32)
            model: PoseModel | None = None
            for start in range(0, inputs.clip.frame_count, policy.chunk_frames):
                stop = min(inputs.clip.frame_count, start + policy.chunk_frames)
                receipt_path = directory / f"chunk-{start}-{stop}.json"
                if receipt_path.exists():
                    chunk = read_document(
                        layout, annotation_reference(layout, receipt_path), PoseChunk
                    )
                    if (chunk.identity, chunk.start, chunk.stop) != (
                        identity,
                        start,
                        stop,
                    ):
                        raise ValueError(
                            "Pose chunk belongs to different input or review"
                        )
                    with np.load(
                        verify_reference(layout, chunk.data), allow_pickle=False
                    ) as archive:
                        values = archive["keypoints"]
                    if (
                        values.shape != output[start:stop].shape
                        or values.dtype != np.float32
                        or not np.isfinite(values).all()
                        or values[~selection["observed"][start:stop]].any()
                    ):
                        raise ValueError("Invalid pose chunk shape/values/missing mask")
                    output[start:stop] = values
                    continue
                for frame in range(start, stop):
                    if not selection["observed"][frame].any():
                        continue
                    if model is None:
                        model = model_factory()
                    output[frame] = infer_selected_frame(
                        frame,
                        inputs.store.read_bgr(inputs.store.row_of(inputs.clip, frame)),
                        selection,
                        model,
                    )
                directory.mkdir(parents=True, exist_ok=True)
                chunk_path = directory / f"chunk-{start}-{stop}.npz"
                write_npz(chunk_path, keypoints=output[start:stop])
                write_document(
                    layout,
                    receipt_path,
                    PoseChunk(
                        identity=identity,
                        start=start,
                        stop=stop,
                        data=annotation_reference(layout, chunk_path),
                    ),
                )
            directory.mkdir(parents=True, exist_ok=True)
            pose_path = directory / "poses.npz"
            write_npz(pose_path, **selection, keypoints=output)
            pose_ref = annotation_reference(layout, pose_path)
        result = PoseResult(
            final_review=final,
            policy=policy_ref,
            identity=identity,
            status=status,
            reason=reason,
            poses=pose_ref,
            reused_from=reuse,
        )
        reference = write_document(layout, directory / "receipt.json", result)
        validate_result(layout, reference)
        return reference
