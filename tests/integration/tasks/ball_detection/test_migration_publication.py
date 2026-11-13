"""Real conversion/native image preparation -> existing publication and learning IO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.tasks.ball_detection.annotation.video import FrameIdentity, write_annotation
from src.tasks.ball_detection.annotation.video.inputs import validate_video_input
from src.tasks.ball_detection.annotation.video.migration import (
    MigrationEvidence,
    convert_v2_annotation,
)
from src.tasks.ball_detection.data.windows.active_ball import ActiveBallWindowDataset
from src.tasks.ball_detection.generate_dataset.frame_store.publishing import (
    publisher as publication,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    MigratedVideoAnnotation,
    MigrationPublishRequest,
    SourceSplit,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.collection.local import retain_local_source
from src.tasks.base.annotation.collection.preparation import prepare_frame_range
from src.tasks.base.annotation.contracts import (
    FrameRange,
    MediaFingerprint,
    PublicationReceipt,
    SourceIdentity,
    content_digest,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import ProductPlan
from tests.support.tasks.ball_detection.v2_annotation import legacy_data
from tests.support.tasks.local_annotation import local_configuration


@pytest.fixture
def migrated(tmp_path: Path) -> tuple[AnnotationLayout, MigrationPublishRequest]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    # Reuse native VFR video generation only. No campaign/resource configuration
    # is registered by this migration route.
    videos = local_configuration(layout).videos
    entries = []
    for video in videos:
        source = video.annotation.source
        expected = video.annotation.model_copy(update=dict(
            frame_range=FrameRange(start=0, stop=5),
            frames=tuple(FrameIdentity(frame_index=i, pts=p) for i, p in enumerate((0, 640, 1920, 2560, 4480))),
        ))
        timeline = validate_video_input(Path(video.path), expected)
        original_bytes = json.dumps(legacy_data(expected)).encode()
        root = layout.path("migration_archive", source.camera_id)
        original = write_asset(layout, root / "v2.json", original_bytes)
        annotation, record = convert_v2_annotation(
            verify_reference(layout, original), expected_input=expected,
            evidence=MigrationEvidence(original_sha256=hashlib.sha256(original_bytes).hexdigest(),
                                       target_semantics="active_ball", target_evidence="original target.scope",
                                       processing_complete=True, processing_evidence="original summary.all_frames_reviewed=true"),
        )
        source_media = retain_local_source(layout, Path(video.path), MediaFingerprint(digest=source.media_sha256, size_bytes=source.size_bytes))
        images = prepare_frame_range(
            layout, SourceIdentity(provider="local", channel_id=content_digest(source.recording_id), video_id=source.sha256,
                                   media=source_media, origin_url=Path(video.path).as_uri()), timeline,
            source_range=expected.frame_range, target_range=FrameRange(start=0, stop=5),
            preparation_digest=expected.sha256, retained_media="jpeg_q90",
        )
        entries.append(MigratedVideoAnnotation(
            expected_input=write_document(layout, root / "input.json", expected),
            original=original, migration=write_document(layout, root / "migration.json", record),
            annotation=write_annotation(layout, root / "v3.json", annotation, expected_input=expected, require_complete=True),
            manifest=images, source_media=source_media,
        ))
    policy = write_document(layout, layout.path("migration_archive", "policy.json"), BallPublicationPolicy(
        jpeg_quality=90, target_semantics="active_ball", source_splits=(SourceSplit(source_id=content_digest("recording-a"), split="val"),),
    ))
    request = MigrationPublishRequest(
        output=ProductPlan(product="frames", dataset_task="ball_detection", dataset_version="migrated-native",
                           component="frames", base_snapshot=None, generation_policy=policy),
        annotations=tuple(sorted(entries, key=lambda x: x.expected_input.digest)),
    )
    return layout, request


def test_migrated_snapshot_replays_after_receipt_interruption(migrated, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, request = migrated
    original_bytes = {e.original: verify_reference(layout, e.original).read_bytes() for e in request.annotations}
    submit, learning = Mock(), Mock()
    monkeypatch.setattr("src.tasks.base.annotation.execution.runner.StageRunner.submit", submit)
    monkeypatch.setattr("src.tasks.base.annotation.learning.service.Learning.register", learning)
    writer = publication.write_document

    def interrupted(layout, path, document):
        if path.name == "receipt.json":
            raise RuntimeError("receipt interruption")
        return writer(layout, path, document)

    with monkeypatch.context() as patch:
        patch.setattr(publication, "write_document", interrupted)
        with pytest.raises(RuntimeError, match="receipt interruption"):
            BallPublisher(layout).publish(request)
    with monkeypatch.context() as patch:
        patch.setattr(BallPublisher, "_build", Mock(side_effect=AssertionError("completed snapshot rebuilt")))
        reference = BallPublisher(layout).publish(request)
        assert BallPublisher(layout).publish(request) == reference
    receipt = read_document(layout, reference, PublicationReceipt)
    snapshot, store = load_snapshot(layout, receipt.snapshot)
    assert len(snapshot.clips) == 2 and len(store) == 10
    assert len(receipt.annotations) == 2 and receipt.adoptions == ()
    assert {c.group_id for c in store.clips} == {content_digest("recording-a")}
    assert {c.split for c in store.clips} == {"val"}
    assert len(store.video_annotations) == 2
    assert len(ActiveBallWindowDataset(store, split="val", window_length=1, stride=1)) == 10
    for clip in store.clips:
        annotation = store.video_annotations[clip.clip_id]
        assert [f.pts for f in annotation.frames] == [0, 640, 1920, 2560, 4480]
        rows = store.clip_rows(clip)
        np.testing.assert_array_equal(store.teacher_masks.coordinate[rows], [1, 1, 1, 0, 1])
        np.testing.assert_array_equal(store.teacher_masks.heatmap[rows], [2, 2, 2, 0, 2])
        assert annotation.frames[1].occlusion == "unknown"
        assert annotation.frames[3].state == "unknown"
        assert store.metadata["clips"][clip.index]["provenance"]["original_v2"] in [e.original.model_dump(mode="json") for e in request.annotations]
    assert all(verify_reference(layout, ref).read_bytes() == data for ref, data in original_bytes.items())
    submit.assert_not_called()
    learning.assert_not_called()


@pytest.mark.parametrize("failure", ["swapped_images", "changed_conversion", "holdout_split"])
def test_migration_rejects_incompatible_images_conversion_or_split(migrated, failure: str) -> None:
    layout, request = migrated
    first, second = request.annotations
    if failure == "swapped_images":
        changed = first.model_copy(update={"manifest": second.manifest})
        bad = request.model_copy(update={"annotations": (changed, second)})
        match = "image manifest"
    elif failure == "changed_conversion":
        path = verify_reference(layout, first.annotation)
        raw = json.loads(path.read_bytes())
        raw["frames"][1]["position"]["x"] = 21.
        different = write_asset(layout, layout.path("migration_archive", "changed.json"), json.dumps(raw).encode())
        changed = first.model_copy(update={"annotation": different})
        bad = request.model_copy(update={"annotations": (changed, second)})
        match = "retained v2 conversion"
    else:
        base_request = request.model_copy(update={"annotations": (first,)})
        receipt = read_document(layout, BallPublisher(layout).publish(base_request), PublicationReceipt)
        policy = write_document(layout, layout.path("migration_archive", "train-policy.json"), BallPublicationPolicy(
            jpeg_quality=90, target_semantics="active_ball", source_splits=(SourceSplit(source_id=content_digest("recording-a"), split="train"),),
        ))
        bad = request.model_copy(update=dict(annotations=(second,), output=request.output.model_copy(update=dict(
            base_snapshot=receipt.snapshot, dataset_version="bad-holdout", generation_policy=policy,
        ))))
        match = "split"
    with pytest.raises(ValueError, match=match):
        BallPublisher(layout).publish(bad)
