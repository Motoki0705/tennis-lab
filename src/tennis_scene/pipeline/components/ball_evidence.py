"""Persistent, pre-gate evidence for the camera-local 2D ball refiner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallPrediction,
)
from src.utils.geometry.keypoints import denormalize_grid_keypoints

if TYPE_CHECKING:
    from src.tennis_scene.pipeline.batching import SourceMemory
    from src.tennis_scene.pipeline.contracts import SourceVideo


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


class BallWindowEvidence:
    """Copy each selected frame atomically. No reference to a batch is retained."""

    def __init__(self, video: SourceVideo, *, aggregation: str, tail_policy: str,
                 memory: SourceMemory | None = None) -> None:
        self.video, self.aggregation, self.tail_policy, self.memory = video, aggregation, tail_policy, memory
        if memory is not None:
            memory.reserve(video.num_frames * 32)
        self.starts: NDArray[np.int64] = np.full(video.num_frames, -1, dtype=np.int64)
        self.times: NDArray[np.int64] = np.empty(video.num_frames, dtype=np.int64)
        self.config: BallCandidateConfig | None = None
        self.arrays: dict[str, NDArray] = {}

    def add(self, frames: Sequence[int], start: int, prediction: BallPrediction, batch_row: int) -> None:
        candidates = prediction.candidates
        values = {name: tensor[batch_row].numpy() for name, tensor in {
            "uv": prediction.coords, "score": prediction.confidence, "heatmaps": prediction.heatmaps,
            "candidate_uv": candidates.coords, "candidate_scores": candidates.scores,
            "candidate_valid": candidates.valid, "candidate_cells": candidates.cells,
            "patches": candidates.patches, "patch_valid": candidates.patch_valid,
        }.items()}
        if any(len(array) != len(frames) for array in values.values()):
            raise ValueError("Ball prediction does not preserve the window's time axis")
        if self.config is None:
            self.config = candidates.config
            if self.memory is not None:
                # Selected arrays, plus the largest grid/candidate validation
                # arrays and restored pixel coordinates at publication.
                size = sum(value[0].nbytes for value in values.values())
                validation = values["heatmaps"][0].size + values["patches"][0].size * 16
                self.memory.reserve(self.video.num_frames * (int(size + validation) + 128))
            self.arrays = {name: np.empty((self.video.num_frames, *value.shape[1:]), dtype=value.dtype)
                           for name, value in values.items()}
        elif self.config != candidates.config or any(
            (value.shape[1:], value.dtype) != (self.arrays[name].shape[1:], self.arrays[name].dtype)
            for name, value in values.items()
        ):
            raise ValueError("Ball candidate config or grid changed within a source video")

        for time, frame in enumerate(frames):
            old_start = self.starts[frame]
            if old_start >= 0:
                if self.aggregation == "nearest_window_centre_then_earlier_start":
                    mid = len(frames) - 1
                    if (abs(2 * time - mid), start) >= (abs(2 * self.times[frame] - mid), old_start):
                        continue
                elif self.aggregation == "max_score" and not values["score"][time] >= self.arrays["score"][frame]:
                    continue
            self.starts[frame], self.times[frame] = start, time
            for name, value in values.items():
                self.arrays[name][frame] = value[time]

    def finish(self) -> tuple[NDArray[np.float32], NDArray[np.float32], BallHeatmapEvidence]:
        covered = int(np.count_nonzero(self.starts >= 0))
        if covered != self.video.num_frames or self.config is None:
            raise ValueError(f"Ball detector covered {covered} of {self.video.num_frames} frames of {self.video.camera_id}; "
                             f"tail_policy={self.tail_policy!r} cannot produce the complete source timeline")
        arrays = self.arrays
        size = self.video.width, self.video.height
        evidence = BallHeatmapEvidence(
            source_size_wh=size, config=self.config, heatmaps=arrays["heatmaps"],
            candidate_uv_px=denormalize_grid_keypoints(arrays["candidate_uv"], *size),
            candidate_scores=arrays["candidate_scores"], candidate_valid=arrays["candidate_valid"],
            candidate_cells=arrays["candidate_cells"], patches=arrays["patches"], patch_valid=arrays["patch_valid"],
            selected_window_start=self.starts, selected_time_index=self.times,
        )
        return arrays["uv"], arrays["score"], evidence
