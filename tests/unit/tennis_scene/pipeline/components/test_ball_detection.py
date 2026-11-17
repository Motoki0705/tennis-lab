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
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_ball_config,
    make_x3d_ball_config,
    materialize_model_weights,
)


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
    predictor = _TypedBallPredictor()

    coords, confidence, evidence = module._predict_video(SourceVideo("cam", tmp_path / "unused.mp4", "hash", 2, 30., 6, 4), predictor)

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
    materialize_model_weights(module)
    monkeypatch.setattr(module, "_load_predictor", _TypedBallPredictor)
    output = module.process(BallDetectionInput(SourceVideo("near", tmp_path / "near.mp4", "hash", num_frames, 30.0, 6, 4)))
    assert isinstance(output, BallDetectionOutput)
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
    predictor = CapturingPredictor()
    module._predict_video(SourceVideo("cam", tmp_path / "unused.mp4", "hash", 2, 30., 6, 4), predictor)


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
    predictor = OverlappingPredictor()
    coords, scores, evidence = module._predict_video(SourceVideo("cam", tmp_path / "clip.mp4", "hash", 3, 30., 6, 4), predictor)
    assert scores[1] == pytest.approx(expected)
    assert evidence.candidate_scores[1, 0] == pytest.approx(expected)
    assert evidence.heatmaps[1].max() == pytest.approx(expected)
    assert evidence.patches[1, 0, 2, 2] == pytest.approx(expected)
    np.testing.assert_allclose(evidence.candidate_uv_px[1, 0], coords[1] * [5, 3])
    assert evidence.selected_window_start[1] == (0 if policy == "max_score" else 1)
    assert evidence.selected_time_index[1] == (1 if policy == "max_score" else 0)


def test_stride_cannot_leave_silent_holes(tmp_path: Path) -> None:
    module = BallDetectionModule(replace(make_ball_config(tmp_path), window_stride=3))
    predictor = _TypedBallPredictor()
    with pytest.raises(ValueError, match="cover every frame"):
        module._predict_video(SourceVideo("cam", tmp_path / "clip.mp4", "hash", 5, 30., 6, 4), predictor)


def test_checkpoint_normalization_mismatch_is_rejected_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.model_io.normalization import BallImageNormalization

    fake = SimpleNamespace(image_normalization=BallImageNormalization())
    monkeypatch.setattr(ball_component.BallDetectionPredictor, "load_from_checkpoint", lambda *args, **kwargs: fake)
    module = BallDetectionModule(replace(make_ball_config(tmp_path), normalize_imagenet=True))
    with pytest.raises(ValueError, match="does not match the saved checkpoint"):
        module._load_predictor()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_centre_policy_matches_training_owners_without_score_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int,
) -> None:
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
    predictor = Predictor()
    coords, scores, evidence = module._predict_video(SourceVideo("cam", tmp_path / "unused", "hash", 11, 30., 6, 4), predictor)
    starts = (0, 3, 6, 7)  # stride 3 with an irregular final backfill
    # Independent oracle: nearest window centre, ties to the earlier start.
    centres = np.asarray(starts) + 1.5
    distance = np.abs(np.arange(11)[:, None] - centres[None])
    distance[(np.arange(11)[:, None] < starts) | (np.arange(11)[:, None] >= np.asarray(starts) + 4)] = np.inf
    expected = np.asarray(starts)[np.argmin(distance, axis=1)]
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
    predictor = _TypedBallPredictor()
    with pytest.raises(ValueError, match="short-clip padding is forbidden"):
        module._predict_video(SourceVideo("cam", tmp_path / "unused", "hash", frames, 30., 6, 4), predictor)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_x3d_windows_keep_native_rgb_and_reset_mdd_per_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int) -> None:
    from src.tasks.ball_detection.preprocessing.rgb_mdd import RGBToRGBMDD
    from src.utils.video.transforms import BgrToRgbUint8Transform

    # Spatial resize is tested separately; keep this owner/MDD test tiny.
    monkeypatch.setattr(ball_component, "BgrToRgbUint8Transform", lambda **_: BgrToRgbUint8Transform(image_size=(4, 6)))
    packets = [FramePacket(index=i, frame=np.full((4, 6, 3), i, np.uint8), original_size=(6, 4)) for i in range(69)]
    monkeypatch.setattr(ball_component, "OpenCVVideoFrameReader", lambda *a, **kw: packets)
    seen = []

    class Predictor:
        configured_frames = 32

        def predict(self, images: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
            assert images.dtype == torch.uint8
            features = RGBToRGBMDD()(images)
            assert features[:, 3:, 0].count_nonzero() == 0
            assert (features[:, 3:, 1] > 0).all()
            torch.testing.assert_close(features[:, :3], images.float().transpose(1, 2) / 255)
            maps = torch.full((len(images), 32, 6, 10), .01)
            for b, first in enumerate(images[:, 0, 0, 0, 0].tolist()):
                seen.append(first)
                maps[b, :, 3, 2] = (first + 50) / 100
            return _prediction(maps, candidate_config)

    module = BallDetectionModule(replace(make_x3d_ball_config(tmp_path), batch_size=batch_size))
    predictor = Predictor()
    uv, scores, evidence = module._predict_video(SourceVideo("cam", tmp_path / "unused", "hash", 69, 30., 43, 31), predictor)
    assert seen == [0, 16, 32, 37]
    # Independent explicit owner intervals, including a centre-distance tie and tail.
    expected: NDArray[np.int64] = np.repeat([0, 16, 32, 37], [24, 16, 11, 18])
    np.testing.assert_array_equal(evidence.selected_window_start, expected)
    np.testing.assert_array_equal(evidence.selected_time_index, np.arange(69) - expected)
    np.testing.assert_allclose(scores, (expected + 50) / 100)
    np.testing.assert_allclose(uv, np.tile([2 / 9, 3 / 5], (69, 1)))
    np.testing.assert_allclose(evidence.candidate_uv_px[:, 0], np.tile([2 / 9 * 42, 3 / 5 * 30], (69, 1)))
    np.testing.assert_array_equal(evidence.candidate_scores[:, 0], scores)


@pytest.mark.parametrize("changes", [dict(model_type="auto"), dict(input_channels=2), dict(num_frames=16),
    dict(image_size=(1280, 720)), dict(normalize_imagenet=True), dict(window_stride=32),
    dict(tail_policy="drop"), dict(overlap_aggregation="max_score"), dict(checkpoint_strict=False),
    dict(checkpoint_weights_only=False), dict(precision="fp16")])
def test_x3d_settings_disagreement_is_rejected_before_load(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(make_x3d_ball_config(tmp_path), **changes)


def test_x3d_explicit_loader_does_not_try_convnext_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    def fail(*args, **kwargs):
        called.append("x3d")
        raise ValueError("bad native artifact")
    def forbidden(*args, **kwargs):
        pytest.fail("ConvNeXt loader must not run for x3d_csn_dpt")
    monkeypatch.setattr(ball_component.X3DDPTPredictor, "load_from_checkpoint", fail)
    monkeypatch.setattr(ball_component.BallDetectionPredictor, "load_from_checkpoint", forbidden)
    module = BallDetectionModule(make_x3d_ball_config(tmp_path))
    with pytest.raises(ValueError, match="bad native artifact"):
        module._load_predictor()
    assert called == ["x3d"]


@pytest.mark.parametrize("precision,dtype", [("fp32", "float32"), ("bf16", "bfloat16")])
def test_x3d_runtime_factory_preserves_precision_and_returns_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str, dtype: str,
) -> None:
    predictor = object()
    calls = []

    def load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return predictor

    monkeypatch.setattr(ball_component.X3DDPTPredictor, "load_from_checkpoint", load)
    config = replace(make_x3d_ball_config(tmp_path), precision=precision)
    spec = BallDetectionModule(config).model_specs()["ball"]
    assert spec.dtype == dtype
    assert spec.architecture == "x3d_csn_dpt"
    assert spec.factory() is predictor
    assert len(calls) == 1
    assert calls[0][0] == config.checkpoint
    assert calls[0][1]["precision"] == precision
    assert calls[0][1]["compile_mode"] == config.compile_mode


def test_fixed_presence_threshold_is_inclusive(tmp_path: Path) -> None:
    base = make_x3d_ball_config(tmp_path)
    threshold = np.float32(.5380124449729919)
    module = BallDetectionModule(replace(base, score_threshold=float(threshold),
                                         trajectory_gate=replace(base.trajectory_gate, enabled=False)))
    scores = np.asarray([np.nextafter(threshold, np.float32(0)), threshold, np.nextafter(threshold, np.float32(1))])
    uv, score, observed = module._accept_detections(np.full((3, 2), 7, np.float32), scores)
    np.testing.assert_array_equal(observed, [False, True, True])
    np.testing.assert_array_equal(uv[:, 0], [0, 7, 7])
    assert score[0] == 0 and score[1] == threshold
