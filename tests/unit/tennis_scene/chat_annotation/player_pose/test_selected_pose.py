from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.person_tracking.features import FeatureExtractor, UnpromptedEncoder
from src.tasks.person_tracking.sequence import (
    DatasetTrackingConfig,
    TrackingConfig,
    track_sequence,
)
from src.tasks.person_tracking.strongsort import StrongSort, StrongSortConfig
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tennis_scene.chat_annotation.player_pose import orchestrator, selected_pose
from src.tennis_scene.chat_annotation.player_pose.reviews import validate_and_remap
from tests.unit.tennis_scene.chat_annotation.player_pose.test_campaign import (
    decision,
    raw_tracks,
    segment,
)


def test_only_selected_real_player_rows_reach_pose() -> None:
    raw = raw_tracks()
    raw.pop("keypoints")
    raw["track_ids"] = np.array([3, 8, 9])
    raw["boxes"] = np.tile(raw["boxes"][:1], (3, 1, 1))
    raw["detection_rows"] = np.array([[0, 3, -1, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
    review = decision(
        [
            segment(3, 0, 4, "player_1"),
            segment(8, 0, 2, "player_1", "duplicate"),
            segment(8, 2, 3, None, "non_player"),
            segment(8, 3, 4, "player_1", "duplicate"),
            segment(9, 0, 4, None, "non_player"),
        ]
    )
    selection = validate_and_remap(
        raw, review, clip_id="clip", raw_hash="hash", required_sheets=["frames.jpg"]
    )
    calls = []

    class Pose:
        def predict(self, request: Pose2DFrameSequenceRequest) -> Pose2DResult:
            calls.extend(request.frame_indices.tolist())
            return Pose2DResult(torch.full((len(request.frame_indices), 17, 3), 1.2))

    output = np.stack(
        [
            selected_pose.infer_selected_frame(
                f, np.zeros((32, 32, 3), np.uint8), selection, Pose()
            )
            for f in range(4)
        ]
    )
    assert calls == [0, 1, 3]
    assert not output[2].any()
    assert selection["detection_rows"][:, 0].tolist() == [0, 3, -1, 9]
    assert selection["raw_track_ids"][:, 0].tolist() == [3, 3, -1, 3]
    assert "keypoints" not in selection


def test_pose_free_features_and_tracking_never_compute_poses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.person_tracking import strongsort

    class Encoder:
        name = "fixture"
        input_size = (16, 8)

        def embed(self, crops: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0]]).repeat(len(crops), 1)

    monkeypatch.setattr(
        strongsort,
        "local_pose",
        lambda *_: pytest.fail("pose computation is forbidden"),
    )
    extractor = FeatureExtractor(None, UnpromptedEncoder(Encoder(), 2))
    frames = [
        extractor.extract(
            f,
            np.zeros((100, 100, 3), np.uint8),
            np.array([f], np.int64),
            np.array([[10, 10, 40, 90]], np.float32),
            np.ones(1, np.float32),
        )
        for f in range(5)
    ]
    assert all(frame.poses is None for frame in frames)
    af = AFLink.__new__(AFLink)
    monkeypatch.setattr(
        af, "links", lambda boxes, seen: ({i: i for i in range(len(boxes))}, [])
    )
    result = track_sequence(frames, fps=30, config=DatasetTrackingConfig(), aflink=af)
    assert result.evidence.poses is None
    assert result.evidence.detection_rows.tolist() == [[-1, -1, 2, 3, 4]]
    assert TrackingConfig().online_config().pose_weight == 0.15
    with pytest.raises(ValueError, match="profile"):
        track_sequence(frames, fps=30, config=TrackingConfig(), aflink=af)
    with pytest.raises(ValueError, match="requires inferred"):
        StrongSort(StrongSortConfig(pose_weight=0.15)).update(frames[0])
    with pytest.raises(ValueError, match="profile"):
        track_sequence(
            [replace(frames[0], poses=np.zeros((1, 17, 3), np.float32))],
            fps=30,
            config=DatasetTrackingConfig(),
            aflink=af,
        )


def test_enqueue_recovers_after_admission_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    campaign = tmp_path / "campaign"
    queue = tmp_path / "queue"
    token = hashlib.sha256(str(campaign.resolve()).encode()).hexdigest()[:16]
    name = f"123_i988-{token}-pose-00003-a0.job"
    (queue / "done").mkdir(parents=True)
    (queue / "done" / name).write_text("fixture")
    config = {
        "assets": {"dino_extension": "/extension.so"},
        "project_root": str(tmp_path),
        "python": "python",
        "queue_dir": str(queue),
    }
    monkeypatch.setattr(
        orchestrator.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("do not enqueue twice"),
    )
    assert orchestrator.enqueue_clip(config, campaign, 3, 0, "pose") == name
