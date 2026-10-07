from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tennis_scene.chat_annotation.player_pose import generation, selected_pose
from src.tennis_scene.chat_annotation.player_pose.reviews import accept_review
from src.tennis_scene.chat_annotation.player_pose.selection import initialize
from src.tennis_scene.chat_annotation.player_pose.storage import (
    clip_root,
    digest,
    read_json,
    write_json,
    write_npz,
)
from tests.unit.tennis_scene.chat_annotation.player_pose.test_campaign import (
    decision,
    raw_tracks,
    segment,
)
from tests.unit.tennis_scene.chat_annotation.player_pose.test_campaign import (
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


def test_full_tracking_generation_does_not_load_vitpose(
    store: BallFrameStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from src.submodules.models.dino import extension, person_detector
    from src.submodules.models.vitpose import pose2d
    from src.tasks.person_tracking import strongsort_offline
    from src.tasks.player_association.appearance import encoders

    campaign = campaign_fixture(store, tmp_path)
    queue_fixture(monkeypatch)
    monkeypatch.setattr(
        extension, "validate_dino_extension", lambda: tmp_path / "weight"
    )
    monkeypatch.setattr(
        pose2d,
        "ViTPosePose2D",
        lambda *_a, **_k: pytest.fail("tracking must not load pose"),
    )

    class Detector:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def predict(self, request: object) -> Any:
            return SimpleNamespace(
                boxes_xyxy=np.array([[2, 2, 15, 20]], np.float32),
                scores=np.array([0.9], np.float32),
            )

        def unload(self) -> None:
            pass

    class Encoder:
        name = "fixture"
        input_size = (16, 8)

        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def embed(self, crops: torch.Tensor) -> torch.Tensor:
            values = torch.zeros(len(crops), 1280)
            values[:, 0] = 1
            return values

    af = strongsort_offline.AFLink.__new__(strongsort_offline.AFLink)
    monkeypatch.setattr(
        af, "links", lambda boxes, seen: ({i: i for i in range(len(boxes))}, [])
    )
    monkeypatch.setattr(strongsort_offline, "AFLink", lambda _: af)
    monkeypatch.setattr(person_detector, "DinoPersonDetector", Detector)
    monkeypatch.setattr(encoders, "ClipReIDEncoder", Encoder)
    generation.generate_clip(campaign, 0)
    root = clip_root(campaign, 0)
    record = read_json(root / "generation.json")
    assert record["all_detections"] == 4 and record["pose_crops"] == 0
    with np.load(root / "tracks.npz") as data:
        assert "keypoints" not in data.files
        assert data["detection_rows"].tolist() == [[-1, -1, 2, 3]]
    monkeypatch.setattr(
        person_detector,
        "DinoPersonDetector",
        lambda *_a, **_k: pytest.fail("resume must use committed tracking"),
    )
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
