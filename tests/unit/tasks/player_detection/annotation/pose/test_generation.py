from pathlib import Path

import numpy as np
import pytest
import torch

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tasks.player_detection.annotation.pose import generation, selected_pose
from src.tasks.player_detection.annotation.pose.reviews import accept_review
from src.tasks.player_detection.annotation.pose.selection import initialize
from src.tasks.player_detection.annotation.pose.storage import (
    clip_root,
    digest,
    read_json,
    write_json,
    write_npz,
)
from tests.unit.tasks.player_detection.annotation.pose.test_campaign import (
    decision,
    raw_tracks,
    segment,
)
from tests.unit.tasks.player_detection.annotation.pose.test_campaign import (
    store as store,
)


def campaign_fixture(store: BallFrameStore, tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign"
    asset = tmp_path / "weight"
    asset.write_bytes(b"fixture")
    initialize(
        campaign,
        {
            "store": str(store.directory),
            "dataset": str(tmp_path / "poses"),
            "project_root": str(tmp_path),
            "presence_threshold": 0.4,
            "chunk_frames": 2,
            "cuda_memory_fraction": 0.5,
            "assets": {
                k: str(asset)
                for k in (
                    "dino",
                    "dino_repository",
                    "dino_extension",
                    "vitpose",
                    "clip_reid",
                    "aflink",
                )
            },
        },
    )
    return campaign


def queue_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENNIS_RUN_ID", "fixture")
    monkeypatch.setenv("TENNIS_GPU_RESOURCE", "all")
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_tracking_requires_corrected_input_before_loading_models(
    store: BallFrameStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.submodules.models.dino import person_detector
    from src.submodules.models.vitpose import pose2d
    from src.tasks.player_association.appearance import encoders

    campaign = campaign_fixture(store, tmp_path)
    queue_fixture(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("Missing corrected input must stop before model construction")

    monkeypatch.setattr(person_detector, "DinoPersonDetector", forbidden)
    monkeypatch.setattr(pose2d, "ViTPosePose2D", forbidden)
    monkeypatch.setattr(encoders, "ClipReIDEncoder", forbidden)
    with pytest.raises(ValueError, match="pinned corrected bbox input"):
        generation.generate_clip(campaign, 0)


def test_full_tracking_generation_does_not_load_vitpose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.submodules.models.dino import person_detector
    from src.submodules.models.vitpose import pose2d
    from src.tasks.base.annotation.paths import AnnotationLayout
    from src.tasks.person_tracking import strongsort_offline
    from src.tasks.player_association.appearance import encoders
    from src.tasks.player_detection.annotation.pose.corrected import load_input
    from tests.support.tasks.player_detection.pose_publication import (
        Linker,
        PublicationFixture,
    )

    fixture = PublicationFixture(tmp_path / "annotation", finalize=False)
    inputs = load_input(fixture.env.layout, fixture.input)
    campaign = tmp_path / "campaign"
    weight = fixture.env.layout.repository / "ckpt" / "tracking.pth"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"synthetic tracking weights")
    config = {
        "project_root": str(fixture.env.layout.repository),
        "corrected_inputs": {
            inputs.manifest.clip_id: fixture.input.model_dump(mode="json")
        },
        "assets": {name: str(weight) for name in ("clip_reid", "aflink")},
        "asset_hashes": {name: digest(weight) for name in ("clip_reid", "aflink")},
        "cuda_memory_fraction": 0.5,
        "generation_mode": "review_then_pose.v1",
    }
    monkeypatch.setattr(
        generation,
        "load_campaign",
        lambda _: (
            config,
            {"clips": [{"clip_id": inputs.manifest.clip_id, "selected": True}]},
            inputs.store,
        ),
    )
    monkeypatch.setattr(AnnotationLayout, "production", lambda _: fixture.env.layout)
    queue_fixture(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("corrected tracking must not load a detector or pose model")

    monkeypatch.setattr(person_detector, "DinoPersonDetector", forbidden)
    monkeypatch.setattr(pose2d, "ViTPosePose2D", forbidden)

    class Encoder:
        name = "fixture"
        input_size = (16, 8)

        def embed(self, crops):
            values = torch.zeros(len(crops), 1280)
            values[:, 0] = 1
            return values

    monkeypatch.setattr(encoders, "ClipReIDEncoder", lambda *args: Encoder())
    monkeypatch.setattr(strongsort_offline, "AFLink", lambda _: Linker())
    generation.generate_clip(campaign, 0)
    root = clip_root(campaign, 0)
    record = read_json(root / "generation.json")
    assert record["all_detections"] == 6 and record["pose_crops"] == 0
    assert record["detector_calls"] == 0
    with np.load(root / "tracks.npz") as data:
        assert "keypoints" not in data.files
        rows = data["detection_rows"]
        assert sorted(rows[rows >= 0].tolist()) == list(range(6))
    monkeypatch.setattr(encoders, "ClipReIDEncoder", forbidden)
    generation.generate_clip(campaign, 0)


def test_pose_resume_reuses_only_completed_chunks_and_keeps_review_hash(
    store: BallFrameStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = campaign_fixture(store, tmp_path)
    queue_fixture(monkeypatch)
    root = clip_root(campaign, 0)
    raw = raw_tracks()
    raw.pop("keypoints")
    write_npz(root / "tracks.npz", **raw)
    write_json(
        root / "generation.json", {"files": {"tracks.npz": digest(root / "tracks.npz")}}
    )
    write_json(
        root / "input.json",
        {"shard_sha256": digest(store.directory / SHARDS_DIR / shard_name(0))},
    )
    review = decision([segment(3, 0, 2, "player_1"), segment(8, 2, 4, "player_1")])
    review.update(
        clip_id=store.clips[0].clip_id, raw_tracks_sha256=digest(root / "tracks.npz")
    )
    write_json(root / "review-input.json", review)
    accept_review(campaign, 0, root / "review-input.json", ["frames.jpg"])
    review_hash = digest(root / "review.json")
    calls = []
    fail = True

    class Pose:
        def predict(self, request: Pose2DFrameSequenceRequest) -> Pose2DResult:
            frame = int(request.frame_indices[0])
            calls.append(frame)
            if fail and frame == 2:
                raise RuntimeError("interrupted GPU process")
            return Pose2DResult(torch.full((1, 17, 3), float(frame)))

        def unload(self) -> None:
            pass

    monkeypatch.setattr(selected_pose, "pose_model", lambda _: Pose())
    with pytest.raises(RuntimeError, match="interrupted"):
        selected_pose.generate_selected_pose(campaign, 0)
    assert not (root / "pose.json").exists()
    fail = False
    selected_pose.generate_selected_pose(campaign, 0)
    assert calls == [0, 1, 2, 2, 3]
    assert digest(root / "review.json") == review_hash
    assert read_json(root / "pose.json")["pose_crops"] == 4
    selected_pose.generate_selected_pose(campaign, 0)
    assert calls == [0, 1, 2, 2, 3]
