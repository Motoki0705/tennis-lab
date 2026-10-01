"""Complete store-clip inference, retaining one prediction per source frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallPrediction,
)
from src.utils.video import (
    BgrToTensorTransform,
    FramePacket,
    iter_temporal_batches,
    iter_temporal_windows,
)


class HoldoutPredictor(Protocol):
    @property
    def configured_frames(self) -> int: ...

    def predict(
        self, images: torch.Tensor, *, candidate_config: BallCandidateConfig
    ) -> BallPrediction: ...


@dataclass(frozen=True)
class FramePredictions:
    """Coordinates are source pixels; rejected scores retain their raw argmax."""

    uv: NDArray[np.float32]
    score: NDArray[np.float32]
    candidate_uv: NDArray[np.float32]
    candidate_score: NDArray[np.float32]
    candidate_valid: NDArray[np.bool_]
    window_start: NDArray[np.int64]
    time_index: NDArray[np.int64]

    def __post_init__(self) -> None:
        n = len(self.score)
        if self.candidate_score.ndim != 2:
            raise ValueError("Candidate scores must be (frames, candidates)")
        if (
            self.uv.shape != (n, 2)
            or self.candidate_uv.shape != (*self.candidate_score.shape, 2)
            or self.candidate_score.shape[0] != n
            or self.candidate_valid.shape != self.candidate_score.shape
            or self.window_start.shape != (n,)
            or self.time_index.shape != (n,)
            or self.score.shape != (n,)
        ):
            raise ValueError("Prediction arrays do not share one frame timeline")
        for value in (self.uv, self.score, self.candidate_uv, self.candidate_score):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Predictions must be finite float32 arrays")
        for value in (self.score, self.candidate_score):
            if ((value < 0) | (value > 1)).any():
                raise ValueError("Prediction scores must be in [0, 1]")
        if self.candidate_valid.dtype != np.bool_:
            raise ValueError("Candidate validity must be boolean")
        for value in (self.window_start, self.time_index):
            if value.dtype != np.int64 or (value < 0).any():
                raise ValueError("Window provenance must contain nonnegative int64 values")


def predict_store_clip(
    store: BallFrameStore,
    clip: ClipRecord,
    predictor: HoldoutPredictor,
    *,
    image_size: tuple[int, int],
    stride: int,
    batch_size: int,
    candidates: BallCandidateConfig,
) -> FramePredictions:
    """Backfill tails and select max score (ties: later window/time slot).

    This matches the pipeline's window selection without trajectory gating.
    JPEGs are decoded incrementally, so a long clip does not fill host RAM.
    Native heatmaps/patches are released after each batch; candidate coordinates
    suffice for the threshold-free top-K recall diagnostic.
    """
    if not 1 <= stride <= predictor.configured_frames:
        raise ValueError("Stride must be between 1 and the configured window length")
    transform = BgrToTensorTransform(image_size=image_size, normalize_imagenet=False)
    stream = (
        FramePacket(
            index=i,
            frame=transform(store.read_bgr(store.row_of(clip, i))),
            original_size=(clip.width, clip.height),
        )
        for i in range(clip.frame_count)
    )
    windows = iter_temporal_windows(
        stream, sequence_length=predictor.configured_frames,
        stride=stride, tail_policy="backfill",
    )
    n, k = clip.frame_count, candidates.max_candidates
    uv = np.zeros((n, 2), np.float32)
    scores = np.zeros(n, np.float32)
    candidate_uv = np.zeros((n, k, 2), np.float32)
    candidate_scores = np.zeros((n, k), np.float32)
    valid = np.zeros((n, k), bool)
    starts = np.full(n, -1, np.int64)
    times = np.full(n, -1, np.int64)
    # Decode in stored-image pixels first, then undo the store's resize.
    scale = np.asarray((clip.width - 1, clip.height - 1), np.float32) / clip.scale
    for batch in iter_temporal_batches(windows, batch_size=batch_size):
        prediction = predictor.predict(batch.tensor, candidate_config=candidates)
        for b, window in enumerate(batch.windows):
            for t, frame in enumerate(window.frame_indices):
                score = float(prediction.confidence[b, t])
                if not np.isfinite(score):
                    raise ValueError("Nonfinite confidence from holdout predictor")
                if starts[frame] >= 0 and score < scores[frame]:
                    continue
                uv[frame] = prediction.coords[b, t].numpy() * scale
                scores[frame] = score
                candidate_uv[frame] = prediction.candidates.coords[b, t].numpy() * scale
                candidate_scores[frame] = prediction.candidates.scores[b, t].numpy()
                valid[frame] = prediction.candidates.valid[b, t].numpy()
                starts[frame], times[frame] = window.start_index, t
    if (starts < 0).any():
        raise ValueError(f"Incomplete holdout timeline for {clip.clip_id}")
    return FramePredictions(uv, scores, candidate_uv, candidate_scores, valid, starts, times)
