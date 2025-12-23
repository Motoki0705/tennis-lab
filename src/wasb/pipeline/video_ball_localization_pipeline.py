"""Single-video ball localization pipeline (WASB inference).

This module provides a lightweight pipeline for estimating per-frame ball
positions from a single video by combining:
- `src.wasb.inference.WASBPredictor`-compatible streaming predictor
- `src.wasb.inference.TrajectoryCompleter` (optional)

The pipeline only returns coordinates/metadata; video rendering lives in
`src/wasb/scripts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.wasb.inference import CompletionResult, TrajectoryCompleter
from src.wasb.utils.video_extractor import VideoExtractor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StreamingBallPredictor(Protocol):
    def reset_tracker(self) -> None: ...

    def predict(
        self,
        frames: NDArray[np.uint8],
        *,
        frame_indices: list[int] | None = None,
    ) -> dict[str, NDArray]: ...


@dataclass(frozen=True)
class VideoBallLocalizationResult:
    """Per-frame ball localization outputs for a single video."""

    video_path: Path
    width: int
    height: int
    fps: float
    frame_indices: NDArray[np.int64]
    ball_xy_px: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    score: NDArray[np.float32]
    completion: CompletionResult | None = None


class VideoBallLocalizationPipeline:
    """Run WASB detection (batched) + optional trajectory completion."""

    def __init__(
        self,
        predictor: StreamingBallPredictor,
        *,
        completer: TrajectoryCompleter | None = None,
        batch_size: int = 64,
    ) -> None:
        self.predictor = predictor
        self.completer = completer
        self.batch_size = int(batch_size)

    def run(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None = None,
    ) -> VideoBallLocalizationResult:
        video_path = Path(video_path)
        extractor = VideoExtractor(video_path)

        all_xy: list[NDArray[np.float32]] = []
        all_vis: list[NDArray[np.bool_]] = []
        all_score: list[NDArray[np.float32]] = []
        all_indices: list[NDArray[np.int64]] = []

        self.predictor.reset_tracker()

        for frames_rgb, start_idx in extractor.iter_batches(batch_size=self.batch_size):
            if max_frames is not None and start_idx >= max_frames:
                break

            end_idx = start_idx + len(frames_rgb)
            if max_frames is not None and end_idx > max_frames:
                frames_rgb = frames_rgb[: max_frames - start_idx]
                end_idx = max_frames

            frame_indices = list(range(start_idx, end_idx))
            results = self.predictor.predict(frames_rgb, frame_indices=frame_indices)

            all_xy.append(results["ball_xy_px"].astype(np.float32, copy=False))
            all_vis.append(results["visibility"].astype(bool, copy=False))
            all_score.append(results["score"].astype(np.float32, copy=False))
            all_indices.append(results["frame_indices"].astype(np.int64, copy=False))

        if not all_indices:
            empty_xy = np.zeros((0, 2), dtype=np.float32)
            empty_bool = np.zeros((0,), dtype=bool)
            empty_f32 = np.zeros((0,), dtype=np.float32)
            empty_i64 = np.zeros((0,), dtype=np.int64)
            return VideoBallLocalizationResult(
                video_path=video_path,
                width=extractor.width,
                height=extractor.height,
                fps=extractor.fps,
                frame_indices=empty_i64,
                ball_xy_px=empty_xy,
                visibility=empty_bool,
                score=empty_f32,
                completion=None,
            )

        frame_indices_arr = np.concatenate(all_indices, axis=0)
        ball_xy_px = np.concatenate(all_xy, axis=0)
        visibility = np.concatenate(all_vis, axis=0)
        score = np.concatenate(all_score, axis=0)

        completion: CompletionResult | None = None
        if self.completer is not None:
            completion = self.completer.complete(
                xy=ball_xy_px.astype(np.float32, copy=False),
                visibility=visibility.astype(np.bool_, copy=False),
                score=score.astype(np.float32, copy=False),
            )

        return VideoBallLocalizationResult(
            video_path=video_path,
            width=extractor.width,
            height=extractor.height,
            fps=extractor.fps,
            frame_indices=frame_indices_arr,
            ball_xy_px=ball_xy_px,
            visibility=visibility,
            score=score,
            completion=completion,
        )


# Backward-compatible alias for older code within the repo.
SingleVideoBallLocalizationPipeline = VideoBallLocalizationPipeline

