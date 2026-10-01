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
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionOutput,
)
from src.utils.video import FramePacket
from tests.unit.tennis_scene.pipeline.config_factories import make_ball_config


class _TypedBallPredictor:
    configured_frames = 2

    def predict(self, images: torch.Tensor) -> BallPrediction:
        batch_size = images.shape[0]
        coords = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
        confidence = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
        return BallPrediction(
            coords=coords.repeat(batch_size, 1, 1),
            confidence=confidence.repeat(batch_size, 1),
            heatmaps=torch.zeros((batch_size, 2, 2, 3)),
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

    coords, confidence = module._predict_video(Path("unused.mp4"), max_frames=2)

    np.testing.assert_allclose(coords, [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(confidence, [0.6, 0.8])


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
        def predict(self, images: torch.Tensor) -> BallPrediction:
            expected = torch.tensor([255, 128, 64], dtype=torch.float32) / 255
            torch.testing.assert_close(images[0, 0, :, 0, 0], expected)
            return super().predict(images)

    module = BallDetectionModule(replace(make_ball_config(tmp_path), image_size=(4, 6),
                                         normalize_imagenet=expected_normalization))
    module._pipeline = CapturingPredictor()  # type: ignore[assignment]
    module._predict_video(Path("unused.mp4"), max_frames=2)


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
