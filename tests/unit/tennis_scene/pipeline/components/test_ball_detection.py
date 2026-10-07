"""Tests for tennis_scene ball-detection pipeline component."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import src.tennis_scene.pipeline.components.ball_detection as ball_component
from src.tasks.ball_detection.model_io import BallPrediction
from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import BallCandidateConfig
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionOutput,
)
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.utils.data.heatmaps import heatmaps_to_argmax
from src.utils.video import FramePacket
from tests.unit.tennis_scene.pipeline.config_factories import make_ball_config


class _TypedBallPredictor:
    configured_frames = 2

    def predict(self, images: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
        batch_size = images.shape[0]
        heatmaps = torch.full((batch_size, 2, 6, 11), .01)
        heatmaps[:, 0, 1, 1], heatmaps[:, 1, 2, 3] = .6, .8
        heatmaps[:, :, 5, 10] = .04
        return _prediction(heatmaps, candidate_config)


def _prediction(heatmaps: torch.Tensor, config: BallCandidateConfig) -> BallPrediction:
    coords, confidence = heatmaps_to_argmax(heatmaps)
    return BallPrediction(
        coords=coords, confidence=confidence, heatmaps=heatmaps,
        candidates=decode_candidates(heatmaps, config=config, subpixel_refine=False),
    )


def test_trajectory_gate_zeroes_rejected_pipeline_frames(tmp_path) -> None:
    frame: NDArray[np.float32] = np.arange(12, dtype=np.float32)
    uv_px = np.stack([50.0 + 20.0 * frame, np.full(12, 120.0, dtype=np.float32)], axis=1).astype(np.float32)
    uv_px[6, 0] += 180.0
    module = BallDetectionModule(make_ball_config(tmp_path))

    gated_px, score, observed = module._accept_detections(uv_px, np.full(12, 0.9, dtype=np.float32))

    assert not bool(observed[6]) and observed[[5, 7]].all()
    np.testing.assert_array_equal(gated_px[6], np.zeros(2, dtype=np.float32))
    assert score[6] == 0.0


def test_predict_video_consumes_typed_task_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packets: list[FramePacket] = [
        FramePacket(
            index=index,
            frame=np.zeros((4, 6, 3), dtype=np.uint8),
            original_size=(6, 4),
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        ball_component,
        "OpenCVVideoFrameReader",
        lambda _path, *, max_frames: packets[:max_frames],
    )
    config = replace(make_ball_config(tmp_path), image_size=(4, 6))
    module = BallDetectionModule(config)
    module._pipeline = _TypedBallPredictor()  # type: ignore[assignment]

    coords, confidence, evidence = module._predict_video(SourceVideo("cam", tmp_path / "unused.mp4", "hash", 2, 30., 6, 4))

    np.testing.assert_allclose(coords, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(confidence, [0.6, 0.8])
    np.testing.assert_allclose(evidence.candidate_scores[:, :2], [[.6, .04], [.8, .04]])


def _process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, num_frames: int, tail_policy: str = "backfill") -> BallDetectionOutput:
    from src.tennis_scene.pipeline.components.ball_detection import BallDetectionInput
    from src.tennis_scene.pipeline.contracts import SourceVideo

    packets = [FramePacket(index=i, frame=np.zeros((4, 6, 3), dtype=np.uint8), original_size=(6, 4)) for i in range(num_frames)]
    monkeypatch.setattr(ball_component, "OpenCVVideoFrameReader", lambda _path, *, max_frames: packets[:max_frames])
    base = make_ball_config(tmp_path)
    module = BallDetectionModule(replace(base, image_size=(4, 6), score_threshold=0.7, tail_policy=tail_policy,
                                         trajectory_gate=replace(base.trajectory_gate, enabled=False)))
    monkeypatch.setattr(module, "load", lambda: setattr(module, "_pipeline", _TypedBallPredictor()))
    output = module.process(BallDetectionInput(SourceVideo("near", tmp_path / "near.mp4", "hash", num_frames, 30.0, 6, 4)))
    assert not module.is_loaded
    return output


def test_process_exposes_one_unidentified_observation_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = _process(tmp_path, monkeypatch, num_frames=2)
    assert output.camera_id == "near" and output.frame_indices.tolist() == [0, 1]
    np.testing.assert_array_equal(output.observed, [False, True])
    np.testing.assert_array_equal(output.uv_px[0], [0, 0])
    np.testing.assert_allclose(output.uv_px[1], [0.3 * 5, 0.4 * 3])  # (W-1, H-1) grid normalization
    np.testing.assert_allclose(output.confidence, [0.0, 0.8])
    assert output.point_kind.tolist() == [0, 1] and output.score_semantics == "model_score"
    assert output.evidence is not None
    np.testing.assert_allclose(output.evidence.candidate_scores[:, 0], [.6, .8])
    np.testing.assert_allclose(output.evidence.candidate_uv_px[0, 0], [.1 * 5, .2 * 3])
    assert output.evidence.candidate_valid[0, 0]  # survives observation threshold
    assert output.evidence.heatmaps[0].max() == np.float32(.6)


def test_incomplete_timeline_stops_instead_of_padding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="covered 2 of 3 frames"):
        _process(tmp_path, monkeypatch, num_frames=3, tail_policy="drop")


@pytest.mark.parametrize("expected_normalization", [False, True])
def test_video_to_predictor_is_always_raw_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected_normalization: bool,
) -> None:
    frame = np.tile(np.array([64, 128, 255], dtype=np.uint8), (4, 6, 1))
    packets = [FramePacket(index=i, frame=frame, original_size=(6, 4)) for i in range(2)]
    monkeypatch.setattr(ball_component, "OpenCVVideoFrameReader", lambda *args, **kwargs: packets)

    class CapturingPredictor(_TypedBallPredictor):
        def predict(self, images: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
            expected = torch.tensor([255, 128, 64], dtype=torch.float32) / 255
            torch.testing.assert_close(images[0, 0, :, 0, 0], expected)
            return super().predict(images, candidate_config=candidate_config)

    module = BallDetectionModule(replace(make_ball_config(tmp_path), image_size=(4, 6),
                                         normalize_imagenet=expected_normalization))
    module._pipeline = CapturingPredictor()  # type: ignore[assignment]
    module._predict_video(SourceVideo("cam", tmp_path / "unused.mp4", "hash", 2, 30., 6, 4))


@pytest.mark.parametrize("policy,expected", [("max_score", .9), ("last_window_wins", .4)])
def test_overlapping_windows_select_all_evidence_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str, expected: float,
) -> None:
    packets = [FramePacket(index=i, frame=np.zeros((4, 6, 3), np.uint8), original_size=(6, 4)) for i in range(3)]
    monkeypatch.setattr(ball_component, "OpenCVVideoFrameReader", lambda *args, **kwargs: packets)

    class OverlappingPredictor(_TypedBallPredictor):
        def predict(self, images: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
            # Regular window [0,1], backfilled window [1,2].
            maps = torch.full((2, 2, 7, 7), .01)
            maps[0, :, 2, 2] = .9
            maps[1, :, 4, 4] = .4
            return _prediction(maps, candidate_config)

    config = replace(make_ball_config(tmp_path), image_size=(4, 6), overlap_aggregation=policy)
    module = BallDetectionModule(config)
    module._pipeline = OverlappingPredictor()  # type: ignore[assignment]
    coords, scores, evidence = module._predict_video(SourceVideo("cam", tmp_path / "clip.mp4", "hash", 3, 30., 6, 4))
    assert scores[1] == pytest.approx(expected)
    assert evidence.candidate_scores[1, 0] == pytest.approx(expected)
    assert evidence.heatmaps[1].max() == pytest.approx(expected)
    assert evidence.patches[1, 0, 2, 2] == pytest.approx(expected)
    np.testing.assert_allclose(evidence.candidate_uv_px[1, 0], coords[1] * [5, 3])
    assert evidence.selected_window_start[1] == (0 if policy == "max_score" else 1)
    assert evidence.selected_time_index[1] == (1 if policy == "max_score" else 0)


def test_stride_cannot_leave_silent_holes(tmp_path: Path) -> None:
    module = BallDetectionModule(replace(make_ball_config(tmp_path), window_stride=3))
    module._pipeline = _TypedBallPredictor()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="cover every frame"):
        module._predict_video(SourceVideo("cam", tmp_path / "clip.mp4", "hash", 5, 30., 6, 4))


def test_checkpoint_normalization_mismatch_is_rejected_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.model_io.normalization import BallImageNormalization

    fake = SimpleNamespace(image_normalization=BallImageNormalization())
    monkeypatch.setattr(ball_component.BallDetectionPredictor, "load_from_checkpoint", lambda *args, **kwargs: fake)
    module = BallDetectionModule(replace(make_ball_config(tmp_path), normalize_imagenet=True))
    with pytest.raises(ValueError, match="does not match the saved checkpoint"):
        module.load()
    assert not module.is_loaded


@pytest.mark.parametrize("batch_size", [1, 3])
def test_centre_policy_matches_training_owners_without_score_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int,
) -> None:
    from src.tasks.ball_refiner_3d.inference.windowing import (
        window_owners,
        window_starts,
    )

    packets = [FramePacket(index=i, frame=np.full((4, 6, 3), i, np.uint8), original_size=(6, 4)) for i in range(11)]
    monkeypatch.setattr(ball_component, "OpenCVVideoFrameReader", lambda *args, **kwargs: packets)

    class Predictor:
        configured_frames = 4

        def predict(self, images: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
            maps = torch.zeros(len(images), 4, 7, 7)
            for b in range(len(images)):
                start = round(float(images[b, 0, 0, 0, 0]) * 255)
                # Later windows have larger scores, but centre selection ignores scores.
                maps[b, :, 2, 2] = (start + 1) / 10
            return _prediction(maps, candidate_config)

    config = replace(make_ball_config(tmp_path), image_size=(4, 6), window_stride=3, batch_size=batch_size,
                     overlap_aggregation="nearest_window_centre_then_earlier_start")
    module = BallDetectionModule(config)
    module._pipeline = Predictor()  # type: ignore[assignment]
    coords, scores, evidence = module._predict_video(SourceVideo("cam", tmp_path / "unused", "hash", 11, 30., 6, 4))
    starts = window_starts(11, 4, 3)
    assert starts == (0, 3, 6, 7)  # irregular final backfill
    expected = np.asarray(starts)[window_owners(11, starts, 4)]
    np.testing.assert_array_equal(evidence.selected_window_start, expected)
    np.testing.assert_array_equal(evidence.selected_time_index, np.arange(11) - expected)
    np.testing.assert_allclose(scores, (expected + 1) / 10)
    np.testing.assert_array_equal(evidence.candidate_scores[:, 0], scores)
    np.testing.assert_array_equal(evidence.heatmaps.max(axis=(1, 2)), scores)
    np.testing.assert_array_equal(evidence.patches[:, 0, 2, 2], scores)
    np.testing.assert_allclose(evidence.candidate_uv_px[:, 0], coords * [5, 3])
    assert expected[8] == 6  # equal centre distance retains earlier window


@pytest.mark.parametrize("frames,tail", [(1, "backfill"), (4, "drop")])
def test_centre_policy_rejects_padding_and_drop(tmp_path: Path, frames: int, tail: str) -> None:
    config = replace(make_ball_config(tmp_path), tail_policy=tail,
                     overlap_aggregation="nearest_window_centre_then_earlier_start")
    module = BallDetectionModule(config)
    module._pipeline = _TypedBallPredictor()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="short-clip padding is forbidden"):
        module._predict_video(SourceVideo("cam", tmp_path / "unused", "hash", frames, 30., 6, 4))
