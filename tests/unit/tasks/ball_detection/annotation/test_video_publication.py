"""Mixed legacy JPEG and new video payloads preserve adopted teachers and RGB."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tasks.ball_detection.annotation.review.contracts import (
    BallEdits,
    BallObservation,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.learning.preparation import (
    prepare_training_windows,
    validate_training_windows,
)
from src.tasks.ball_detection.data.windows.snapshot import snapshot_ball_store
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    PublishRequest,
    SourceSplit,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import PublicationReceipt, content_digest
from tests.support.tasks.ball_detection.annotation import Environment
from tests.support.tasks.ball_detection.annotation_learning import policy, publish


def test_video_publication_hardlinks_media_and_reads_identical_window_inputs(
    tmp_path: Path,
) -> None:
    env, publication, _ = publish(tmp_path, image_storage="video_q90")
    receipt = read_document(env.layout, publication, PublicationReceipt)
    snapshot, store = load_snapshot(env.layout, receipt.snapshot)
    assert store.metadata["schema_version"] == "ball_detection_frames.v3"
    assert len(store.video_frames) == 1
    clip = store.clips[-1]
    path = store.image_path(clip)
    media = env.layout.artifact_path(env.manifest.media)
    assert path.stat().st_ino == media.stat().st_ino
    assert len(list((store.directory / "shards").iterdir())) == 3  # Only old JPEGs.
    frozen, hashes = snapshot_ball_store(
        store.directory, tmp_path / "training-snapshot"
    )
    assert frozen.image_path(clip).stat().st_ino == path.stat().st_ino
    assert hashes[clip.clip_id] == snapshot.clips[-1].shard.digest
    reference = prepare_training_windows(
        env.layout, publication, policy(), env.layout.path("campaigns", "prepared")
    )
    prepared = validate_training_windows(env.layout, reference)
    assert prepared.manifest is not None
    manifest = env.layout.artifact_path(prepared.manifest)
    encoded = FrozenBallWindowDataset(
        manifest, split="train", requires_pose=False, jpeg_decoder="nvjpeg"
    )
    decoded = FrozenBallWindowDataset(
        manifest, split="train", requires_pose=False, jpeg_decoder="opencv"
    )
    index = next(
        i
        for i, (r, _) in enumerate(encoded.windows)
        if encoded.records[r]["clip"]["clip_id"] == clip.clip_id
    )
    left, right = encoded[index], decoded[index]
    packed = left["jpeg"].numpy().tobytes()
    offset = 0
    for position, size in enumerate(left["jpeg_lengths"]):
        payload = packed[offset : offset + size]
        offset += size
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        np.testing.assert_array_equal(
            image[..., ::-1].transpose(2, 0, 1), right["rgb"][position].numpy()
        )
        np.testing.assert_array_equal(
            image, store.read_bgr(store.row_of(clip, position + 1))
        )
    for key in ("uv", "timestamps", "position_valid", "heatmap_valid", "frame_indices"):
        np.testing.assert_array_equal(left[key].numpy(), right[key].numpy())
    assert right["position_valid"].sum() == 32
    assert (
        store.image_path(store.clips[0]).stat().st_ino
        == (tmp_path / "data/ball_detection/base/shards/clip-00000.bin").stat().st_ino
    )
    # Identity is checked again on every read, including changed video bytes.
    with path.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        encoded[index]


def test_video_store_rejects_missing_contract_and_corrupt_frame_hash(
    tmp_path: Path,
) -> None:
    env, publication, _ = publish(tmp_path, image_storage="video_q90")
    receipt = read_document(env.layout, publication, PublicationReceipt)
    _, store = load_snapshot(env.layout, receipt.snapshot)
    clip = store.clips[-1]
    path = store.directory / "metadata.json"
    raw = json.loads(path.read_text())
    raw["video_frames"][clip.clip_id]["frame_sha256"][1] = "0" * 64
    path.write_text(json.dumps(raw))
    altered = BallFrameStore(store.directory)
    with pytest.raises(ValueError, match="JPEG differs"):
        altered.read_jpeg(altered.row_of(clip, 1))
    raw["video_frames"] = {}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="identify existing clips"):
        BallFrameStore(store.directory)


@pytest.mark.parametrize("storage", ["jpeg", "video_q90"])
def test_next_publication_preserves_existing_video_payloads(
    tmp_path: Path, storage: str
) -> None:
    first, publication, _ = publish(tmp_path, image_storage="video_q90")
    previous = read_document(first.layout, publication, PublicationReceipt)
    old_snapshot, old = load_snapshot(first.layout, previous.snapshot)
    env = Environment(
        first.layout.root,
        video_id="second",
        campaign_id="second",
        dataset_version="second",
        actor_attempt="second",
        frame_count=34,
        base_snapshot=previous.snapshot,
        publication_policy=BallPublicationPolicy.model_validate_json(
            json.dumps(
                {
                    "jpeg_quality": 90,
                    "image_storage": storage,
                    "source_splits": [
                        SourceSplit(
                            source_id=content_digest(["local", "second"]), split="train"
                        ).model_dump()
                    ],
                }
            )
        ),
    )
    images = []
    for start in (1, 17):
        indices = tuple(range(start, start + 16))
        image = env.page(indices)
        images.append(image)
        visible = BallObservation(
            track_id="ball", state="visible", center=(16.0, 12.0), origin="manual"
        )
        env.apply(
            BallEdits(
                images=(image,),
                operations=tuple(
                    env.replace(index, image, observations=(visible,))
                    for index in indices
                ),
            )
        )
    adoption = env.adapter.prepare_adoption(
        env.layout,
        env.context,
        env.manifest,
        env.head(),
        (env.telemetry(tuple(images)),),
        reason="second publication retains video identity",
    )
    env.store.seal(env.head(), adoption, env.adapter)
    result = BallPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, product="frames", adoptions=(adoption,))
    )
    receipt = read_document(env.layout, result, PublicationReceipt)
    snapshot, store = load_snapshot(env.layout, receipt.snapshot)
    for before, after in zip(
        old_snapshot.clips, snapshot.clips[: len(old.clips)], strict=True
    ):
        assert before.model_dump(exclude={"shard"}) == after.model_dump(
            exclude={"shard"}
        )
        assert (before.shard.digest, before.shard.size_bytes) == (
            after.shard.digest,
            after.shard.size_bytes,
        )
    for clip in old.clips:
        assert (
            store.image_path(clip).stat().st_ino == old.image_path(clip).stat().st_ino
        )
        assert store.video_frames.get(clip.clip_id) == old.video_frames.get(
            clip.clip_id
        )
    assert store.teacher_masks is not None and old.teacher_masks is not None
    np.testing.assert_array_equal(
        store.teacher_masks.coordinate[: len(old)], old.teacher_masks.coordinate
    )
    np.testing.assert_array_equal(
        store.read_bgr(store.row_of(old.clips[-1], 3)),
        old.read_bgr(old.row_of(old.clips[-1], 3)),
    )
    assert len(store.video_frames) == (2 if storage == "video_q90" else 1)
