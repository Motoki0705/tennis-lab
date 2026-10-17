"""Freeze bbox revisions for ID review; apply returned edits through the same CAS."""

from typing import Literal

from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
    Digest,
    Document,
    FrameMap,
)
from src.tasks.base.annotation.editing import (
    ReviewContext,
    Revision,
    RevisionConflict,
    RevisionStore,
)
from src.tasks.base.annotation.editing.storage import retained_adoption
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage.references import protection

from .contracts import PlayerLabels


class IdReviewFrame(Document):
    frame: FrameMap
    frame_digest: Digest
    observation_ids: tuple[Digest, ...]


class IdReviewBinding(Document):
    schema_version: Literal["player.id-review-input.v1"] = "player.id-review-input.v1"
    context: ArtifactRef
    revision: ArtifactRef
    bbox_review: ArtifactRef
    frames: tuple[IdReviewFrame, ...]


def _frames(labels: PlayerLabels) -> tuple[IdReviewFrame, ...]:
    return tuple(
        IdReviewFrame(
            frame=f.frame,
            frame_digest=f.sha256,
            observation_ids=tuple(o.observation_id for o in f.decision.observations),
        )
        for f in labels.frames
    )


def validate_binding(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    context: ArtifactRef,
    revision: ArtifactRef,
    labels: PlayerLabels,
) -> IdReviewBinding:
    from .adapter import PlayerReviewAdapter

    binding: IdReviewBinding = read_document(layout, reference, IdReviewBinding)
    if (binding.context, binding.revision) != (context, revision):
        raise RevisionConflict("ID review belongs to a different bbox revision/context")
    node = read_document(layout, revision, Revision)
    review = read_document(layout, binding.bbox_review, AdoptionReceipt)
    if (
        node.context != context
        or read_document(layout, node.payload, PlayerLabels) != labels
        or binding.frames != _frames(labels)
        or (review.task, review.clip_id, review.revision, review.policy)
        != ("player", labels.clip_id, revision, labels.policy)
    ):
        raise ValueError(
            "ID review observation/frame/PTS binding differs from bbox input"
        )
    bound = read_document(layout, context, ReviewContext)
    manifest = read_document(layout, bound.manifest, ClipManifest)
    PlayerReviewAdapter().validate_adoption(layout, bound, manifest, revision, review)
    return binding


def freeze_for_id_review(
    store: RevisionStore,
    context: ArtifactRef,
    revision: ArtifactRef,
    bbox_review: ArtifactRef,
) -> ArtifactRef:
    """Keep the reviewed head editable until downstream ID corrections are final."""
    from .adapter import PlayerReviewAdapter

    adapter = PlayerReviewAdapter()
    head, sealed = store.current(context, adapter)
    if head != revision or sealed is not None:
        raise RevisionConflict("ID review requires the current unsealed bbox revision")
    bound = read_document(store.layout, context, ReviewContext)
    _, manifest, _ = store.context(bound, adapter)
    review = read_document(store.layout, bbox_review, AdoptionReceipt)
    adapter.validate_adoption(store.layout, bound, manifest, revision, review)
    node = read_document(store.layout, revision, Revision)
    labels = read_document(store.layout, node.payload, PlayerLabels)
    binding = IdReviewBinding(
        context=context,
        revision=revision,
        bbox_review=bbox_review,
        frames=_frames(labels),
    )
    reference = write_document(
        store.layout,
        store.layout.path(
            "annotations",
            "player",
            manifest.clip_id,
            "id-review",
            binding.sha256 + ".json",
        ),
        binding,
    )
    evidence = retained_adoption(store.layout, revision, bbox_review, adapter)
    store.catalog.protect_once(
        protection(
            store.layout,
            identity="player-id-review-v1-" + reference.digest,
            kind="active",
            reason="bbox revision and image proof retained for downstream ID review",
            roots=(reference,),
            documents=(binding,),
            dependencies=evidence.references,
        )
    )
    return reference


def load_id_review_input(
    store: RevisionStore, reference: ArtifactRef
) -> tuple[IdReviewBinding, PlayerLabels]:
    """Read exact corrected boxes/score origins, refusing stale or sealed input."""
    from .adapter import PlayerReviewAdapter

    binding = read_document(store.layout, reference, IdReviewBinding)
    head, sealed = store.current(binding.context, PlayerReviewAdapter())
    if head != binding.revision or sealed is not None:
        raise RevisionConflict("ID review requires the current unsealed bbox revision")
    node = read_document(store.layout, head, Revision)
    labels = read_document(store.layout, node.payload, PlayerLabels)
    validate_binding(store.layout, reference, binding.context, head, labels)
    return binding, labels
