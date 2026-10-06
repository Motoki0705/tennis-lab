"""CPU coverage for the inference dependencies imported by player-pose generation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import FeatureExtractor
from src.tasks.person_tracking.sequence import TrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort import StrongSort, StrongSortConfig
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tennis_scene.chat_annotation.player_pose import generation


def two_people(index: int, reverse: bool = False) -> DetectionFeatures:
    boxes = np.tile(np.array([[10, 20, 50, 120]], np.float32), (2, 1))
    poses: np.ndarray = np.ones((2, 17, 3), np.float32)
    poses[:, :, :2] = boxes[:, None, :2] + np.array([0.25, 0.75])[:, None, None] * (
        boxes[:, None, 2:] - boxes[:, None, :2]
    )
    poses[:, :, 2] = 0.8
    if reverse:
        poses = poses[::-1].copy()
    return DetectionFeatures(
        index,
        np.arange(2 * index, 2 * index + 2, dtype=np.int64),
        boxes,
        np.full(2, 0.9, np.float32),
        poses,
        np.tile(np.array([[1, 0]], np.float32), (2, 1)),
        np.ones(2, bool),
    )


def test_pose_resolves_identity_switch_and_keeps_source_rows() -> None:
    tracker = StrongSort(StrongSortConfig(pose_weight=0.15))
    for frame in range(3):
        tracker.update(two_people(frame))
    result = tracker.update(two_people(3, reverse=True))
    assert result.track_ids.tolist() == [2, 1]
    assert result.detection_rows.tolist() == [6, 7]


def test_shared_sequence_retains_gaps_without_synthetic_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    af = AFLink.__new__(AFLink)
    monkeypatch.setattr(
        af, "links", lambda boxes, seen: ({i: i for i in range(len(boxes))}, [])
    )
    frames = [two_people(i, reverse=i >= 3) for i in range(7)]
    empty = frames[5]
    frames[5] = replace(
        empty,
        **{
            name: getattr(empty, name)[:0]
            for name in (
                "rows",
                "boxes",
                "scores",
                "poses",
                "embeddings",
                "appearance_valid",
            )
        },
    )
    result = track_sequence(frames, fps=30.0, config=TrackingConfig(), aflink=af)
    assert result.evidence.detection_rows[:, 3].tolist() == [7, 6]
    assert (
        result.reconstruction is not None
        and result.reconstruction.interpolated[:, 5].all()
    )
    assert not result.observed[:, 5].any()
    assert not result.evidence.require_poses()[:, 5].any()


def test_feature_extractor_uses_stream_api_and_preserves_raw_scores() -> None:
    class Pose:
        def predict(self, request: Pose2DFrameSequenceRequest) -> Pose2DResult:
            assert request.frame_indices.tolist() == [5, 5]
            assert list(request.frames_bgr)[0][0] == 5
            values = torch.ones(2, 17, 3)
            values[..., 2] = 1.2
            return Pose2DResult(values)

    class Encoder:
        dimension = 2
        name = "fixture"
        input_size = (16, 8)

        def embed(self, crops: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
            assert (prompts[..., 2] > 1).all()
            return torch.tensor([[1.0, 0.0]]).repeat(len(crops), 1)

    output = FeatureExtractor(Pose(), Encoder()).extract(
        5,
        np.zeros((100, 100, 3), np.uint8),
        np.array([10, 11], np.int64),
        np.array([[10, 10, 40, 90], [50, 10, 80, 90]], np.float32),
        np.ones(2, np.float32),
    )
    assert output.rows.tolist() == [10, 11]
    assert (output.require_poses()[..., 2] > 1).all()


def test_real_generation_import_path_is_available_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ImportsComplete(RuntimeError):
        pass

    monkeypatch.setenv("TENNIS_RUN_ID", "unit-test-import-only")
    monkeypatch.setenv("TENNIS_GPU_RESOURCE", "all")
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda *_: pytest.fail("CPU test must not allocate CUDA"),
    )

    def stop_before_models(_: Path) -> None:
        raise ImportsComplete("all generation dependencies imported")

    monkeypatch.setattr(generation, "load_campaign", stop_before_models)
    with pytest.raises(ImportsComplete):
        generation.generate_clip(tmp_path, 0)
