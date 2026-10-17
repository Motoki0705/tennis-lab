from pathlib import Path

import pytest
import torch

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tasks.player_detection.annotation.player_pose import generation, selected_pose
from src.tasks.player_detection.annotation.player_pose.reviews import accept_review
from src.tasks.player_detection.annotation.player_pose.selection import initialize
from src.tasks.player_detection.annotation.player_pose.storage import (
    clip_root,
    digest,
    read_json,
    write_json,
    write_npz,
)
from tests.unit.tasks.player_detection.annotation.player_pose.test_campaign import (
    decision,
    raw_tracks,
    segment,
)
from tests.unit.tasks.player_detection.annotation.player_pose.test_campaign import (
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
