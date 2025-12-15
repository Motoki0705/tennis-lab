from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.wasb.inference.trajectory_completion import CompletionResult, TrajectoryCompleter
from src.wasb.inference.video_ball_localization import SingleVideoBallLocalizationPipeline


def _write_tiny_video(path: Path, *, num_frames: int = 8, w: int = 64, h: int = 48) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    try:
        for i in range(num_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 10) % 255
            writer.write(frame)
    finally:
        writer.release()


@dataclass
class _DummyPredictor:
    expected_next: int = 0

    def reset_tracker(self) -> None:
        self.expected_next = 0

    def predict(self, frames: np.ndarray, *, frame_indices=None):  # type: ignore[no-untyped-def]
        if frame_indices is None:
            frame_indices = list(range(self.expected_next, self.expected_next + len(frames)))
        self.expected_next = frame_indices[-1] + 1 if frame_indices else self.expected_next

        xy = np.array([[float(i), float(i + 1)] for i in frame_indices], dtype=np.float32)
        vis = np.ones((len(frame_indices),), dtype=bool)
        score = np.full((len(frame_indices),), 0.9, dtype=np.float32)
        idx = np.array(frame_indices, dtype=np.int64)
        return {
            "ball_xy_px": xy,
            "ball_uv": xy.copy(),
            "visibility": vis,
            "score": score,
            "frame_indices": idx,
        }


class _IdentityCompleter(TrajectoryCompleter):
    def complete(self, xy, visibility, score):  # type: ignore[no-untyped-def]
        vis_code = np.where(visibility, 1, 0).astype(np.int32)
        confidence = np.where(visibility, score, 0.0).astype(np.float32)
        return CompletionResult(
            xy=xy.astype(np.float32, copy=False),
            visibility=vis_code,
            confidence=confidence,
            gaps_filled=0,
            outliers_removed=0,
        )


def test_pipeline_runs_and_concatenates_batches(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, num_frames=8)

    predictor = _DummyPredictor()
    pipeline = SingleVideoBallLocalizationPipeline(predictor, batch_size=3)
    result = pipeline.run(video_path)

    assert result.frame_indices.tolist() == list(range(8))
    assert result.ball_xy_px.shape == (8, 2)
    assert bool(result.visibility.all()) is True
    assert np.allclose(result.score, 0.9)


def test_pipeline_applies_completion(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, num_frames=5)

    predictor = _DummyPredictor()
    pipeline = SingleVideoBallLocalizationPipeline(
        predictor, batch_size=2, completer=_IdentityCompleter()
    )
    result = pipeline.run(video_path)

    assert result.completion is not None
    assert result.completion.xy.shape == (5, 2)
    assert result.completion.visibility.tolist() == [1, 1, 1, 1, 1]

