"""Persistent, pre-gate evidence for the camera-local 2D ball refiner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallPrediction,
)
from src.utils.geometry.keypoints import denormalize_grid_keypoints


@dataclass(frozen=True)
class BallHeatmapEvidence:
    """Native maps and local peaks from the same selected window per frame.

    T is the complete source timeline, K/P come from config, H/W are the
    native model output grid. uv_px is in source-video pixels (Wsource-1,
    Hsource-1 grid scaling). cells and patches refer to the native grid;
    patch_valid distinguishes image boundaries from probability zero.
    Scores are uncalibrated sigmoid values, not ball existence probabilities.
    selected_window_start / selected_time_index identify overlap selection,
    including repeated input frames when the clip is shorter than one window.
    """

    source_size_wh: tuple[int, int]
    config: BallCandidateConfig
    heatmaps: NDArray[np.float32]  # T,H,W
    candidate_uv_px: NDArray[np.float32]  # T,K,2
    candidate_scores: NDArray[np.float32]  # T,K
    candidate_valid: NDArray[np.bool_]  # T,K
    candidate_cells: NDArray[np.int64]  # T,K,2 (x,y)
    patches: NDArray[np.float32]  # T,K,P,P
    patch_valid: NDArray[np.bool_]  # T,K,P,P
    selected_window_start: NDArray[np.int64]  # T
    selected_time_index: NDArray[np.int64]  # T

    def __post_init__(self) -> None:
        if len(self.source_size_wh) != 2 or any(type(s) is not int or s <= 0 for s in self.source_size_wh):
            raise ValueError("Ball evidence requires a positive source image size")
        if self.heatmaps.ndim != 3 or min(self.heatmaps.shape) <= 0:
            raise ValueError("Ball evidence requires native heatmaps (T,H,W)")
        t, h, w = self.heatmaps.shape
        k, p = self.config.max_candidates, self.config.patch_size
        shapes = (
            (self.candidate_uv_px, (t, k, 2), np.float32),
            (self.candidate_scores, (t, k), np.float32),
            (self.candidate_valid, (t, k), np.bool_),
            (self.candidate_cells, (t, k, 2), np.int64),
            (self.patches, (t, k, p, p), np.float32),
            (self.patch_valid, (t, k, p, p), np.bool_),
            (self.selected_window_start, (t,), np.int64),
            (self.selected_time_index, (t,), np.int64),
            (self.heatmaps, (t, h, w), np.float32),
        )
        for array, shape, dtype in shapes:
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("Ball evidence shape/dtype disagrees with its grid/config")
        for values in (self.heatmaps, self.candidate_scores, self.patches):
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError("Ball evidence probabilities must be finite in [0,1]")
        if not np.isfinite(self.candidate_uv_px).all() or (self.candidate_uv_px < 0).any() or (self.candidate_uv_px > np.asarray(self.source_size_wh) - 1).any():
            raise ValueError("Ball candidate coordinates lie outside the source image")
        valid, cells = self.candidate_valid, self.candidate_cells
        if (cells < 0).any() or (cells >= [w, h]).any():
            raise ValueError("Ball candidate cells lie outside the native grid")
        source_scale = np.maximum(np.asarray(self.source_size_wh) - 1, 1)
        grid_coords = self.candidate_uv_px / source_scale * [w - 1, h - 1]
        if (np.abs(grid_coords[valid] - cells[valid]) > .5001).any():
            raise ValueError("Ball candidate coordinates disagree with their native peak cells")
        if (valid[:, 1:] & ~valid[:, :-1]).any() or (self.candidate_scores[:, 1:] > self.candidate_scores[:, :-1]).any():
            raise ValueError("Ball candidates must be packed in descending score order")
        if any(np.any(array[~valid]) for array in (self.candidate_uv_px, self.candidate_scores, cells, self.patches, self.patch_valid)):
            raise ValueError("Invalid ball candidate slots must be zero with false masks")
        offset: NDArray[np.int64] = np.arange(p, dtype=np.int64) - p // 2
        x = cells[..., 0, None, None] + offset[None, :]
        y = cells[..., 1, None, None] + offset[:, None]
        expected_mask = valid[..., None, None] & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if not np.array_equal(self.patch_valid, expected_mask):
            raise ValueError("Ball patch mask must identify valid in-grid cells")
        samples = self.heatmaps[np.arange(t)[:, None, None, None], y.clip(0, h - 1), x.clip(0, w - 1)]
        if not np.array_equal(self.patches, np.where(expected_mask, samples, 0.0)):
            raise ValueError("Ball patches must preserve native heatmap samples")
        if not np.array_equal(self.candidate_scores, self.patches[..., p // 2, p // 2]):
            raise ValueError("Ball candidate scores must equal the native centre cell")
        if (self.selected_window_start < 0).any() or (self.selected_window_start > np.arange(t)).any() or (self.selected_time_index < 0).any():
            raise ValueError("Invalid ball evidence window provenance")


@dataclass(frozen=True)
class SelectedBallFrame:
    """An atomic selection: point, map and candidates always use one window."""

    prediction: BallPrediction
    batch_index: int
    time_index: int
    window_start: int

    @property
    def score(self) -> float:
        return float(self.prediction.confidence[self.batch_index, self.time_index])


def assemble_ball_evidence(
    frames: Sequence[SelectedBallFrame], *, source_size_wh: tuple[int, int],
) -> BallHeatmapEvidence:
    """Stack selected task outputs without thresholding, averaging or re-decoding."""
    if not frames:
        raise ValueError("Ball evidence requires at least one predicted frame")
    config = frames[0].prediction.candidates.config
    if any(frame.prediction.candidates.config != config for frame in frames):
        raise ValueError("Ball candidate config changed within a source video")
    coords = np.stack([f.prediction.candidates.coords[f.batch_index, f.time_index].numpy() for f in frames])
    return BallHeatmapEvidence(
        source_size_wh=source_size_wh, config=config,
        heatmaps=np.stack([f.prediction.heatmaps[f.batch_index, f.time_index].numpy() for f in frames]),
        candidate_uv_px=denormalize_grid_keypoints(coords, *source_size_wh),
        candidate_scores=np.stack([f.prediction.candidates.scores[f.batch_index, f.time_index].numpy() for f in frames]),
        candidate_valid=np.stack([f.prediction.candidates.valid[f.batch_index, f.time_index].numpy() for f in frames]),
        candidate_cells=np.stack([f.prediction.candidates.cells[f.batch_index, f.time_index].numpy() for f in frames]),
        patches=np.stack([f.prediction.candidates.patches[f.batch_index, f.time_index].numpy() for f in frames]),
        patch_valid=np.stack([f.prediction.candidates.patch_valid[f.batch_index, f.time_index].numpy() for f in frames]),
        selected_window_start=np.asarray([f.window_start for f in frames], dtype=np.int64),
        selected_time_index=np.asarray([f.time_index for f in frames], dtype=np.int64),
    )
