"""Deterministic synthetic multi-ball scenes and unordered detector candidates."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tasks.ball_tracking.data.types import BallTrackingBatch

FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float32]]
BoolArray: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
IntArray: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]


class SyntheticBallTrackingDataset(Dataset[dict[str, torch.Tensor]]):
    """Generate padded multi-ball clips without leaking GT order to the model."""

    def __init__(self, config: Any, *, split: str) -> None:
        self.config = config
        split_sizes = config.get("split_sizes", {})
        self.num_samples = int(split_sizes[split])
        self.split = split
        self.seed = (
            int(config.seed) + {"train": 0, "val": 100_000, "test": 200_000}[split]
        )
        self.max_frames = int(config.max_frames)
        self.min_frames = int(config.min_frames)
        self.max_views = int(config.max_views)
        self.min_views = int(config.min_views)
        self.max_balls = int(config.max_balls)
        self.min_balls = int(config.min_balls)
        self.max_candidates = int(config.max_candidates)
        self.dropout_probability = float(config.dropout_probability)
        self.false_positive_probability = float(config.false_positive_probability)
        self.duplicate_probability = float(config.duplicate_probability)
        self.uv_noise_std = float(config.uv_noise_std)
        self.num_court_keypoints = int(config.num_court_keypoints)

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _trajectory(rng: np.random.Generator, length: int) -> FloatArray:
        time = np.linspace(0.0, 1.0, length, dtype=np.float32)
        start = np.asarray(
            rng.uniform([0.15, 0.15, 0.08], [0.85, 0.85, 0.35]), dtype=np.float32
        )
        velocity = np.asarray(
            rng.uniform([-0.35, -0.45, 0.20], [0.35, 0.45, 0.65]), dtype=np.float32
        )
        position = start[None] + time[:, None] * velocity[None]
        position[:, 2] += -0.55 * time * time
        position[:, 2] = np.abs(position[:, 2])
        return cast(
            FloatArray, np.asarray(np.clip(position, 0.0, 1.0), dtype=np.float32)
        )

    @staticmethod
    def _project(position: FloatArray, camera_index: int) -> FloatArray:
        angle = 2.0 * np.pi * camera_index / 8.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = (position[..., 0] - 0.5) * cos_a - (position[..., 1] - 0.5) * sin_a
        depth = (
            1.8 + (position[..., 0] - 0.5) * sin_a + (position[..., 1] - 0.5) * cos_a
        )
        u = 0.5 + 0.65 * x / depth
        v = 0.82 - 0.65 * position[..., 2] / depth
        return cast(FloatArray, np.asarray(np.stack([u, v], axis=-1), dtype=np.float32))

    def __getitem__(self, index: int) -> BallTrackingBatch:
        rng = np.random.default_rng(self.seed + index)
        valid_frames = int(rng.integers(self.min_frames, self.max_frames + 1))
        valid_views = int(rng.integers(self.min_views, self.max_views + 1))
        num_balls = int(rng.integers(self.min_balls, self.max_balls + 1))

        position: FloatArray = np.zeros(
            (self.max_frames, self.max_balls, 3), dtype=np.float32
        )
        present: BoolArray = np.zeros((self.max_frames, self.max_balls), dtype=np.bool_)
        clean_uv: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.max_balls, 2), dtype=np.float32
        )
        clean_visible: BoolArray = np.zeros(
            (self.max_views, self.max_frames, self.max_balls), dtype=np.bool_
        )
        for person_index in range(num_balls):
            start = int(rng.integers(0, max(valid_frames // 3, 1)))
            end = int(
                rng.integers(max(start + 2, valid_frames * 2 // 3), valid_frames + 1)
            )
            trajectory = self._trajectory(rng, end - start)
            position[start:end, person_index] = trajectory
            present[start:end, person_index] = True
            for view in range(valid_views):
                projected = self._project(trajectory, view)
                visible = np.logical_and(
                    (projected >= 0.0).all(-1), (projected <= 1.0).all(-1)
                )
                clean_uv[view, start:end, person_index] = projected
                clean_visible[view, start:end, person_index] = visible

        uv: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.max_candidates, 2), dtype=np.float32
        )
        score: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.max_candidates), dtype=np.float32
        )
        candidate_mask = np.zeros_like(score, dtype=np.bool_)
        visible = np.zeros_like(score, dtype=np.bool_)
        candidate_gt_index: IntArray = np.full_like(score, -1, dtype=np.int64)

        coherent_fp = rng.uniform(0.05, 0.95, size=(valid_views, 2)).astype(np.float32)
        coherent_velocity = rng.uniform(-0.015, 0.015, size=(valid_views, 2)).astype(
            np.float32
        )
        for view in range(valid_views):
            for frame in range(valid_frames):
                candidates: list[tuple[np.ndarray, float, bool, int]] = []
                for ball_index in range(num_balls):
                    if not clean_visible[view, frame, ball_index]:
                        continue
                    if rng.random() < self.dropout_probability:
                        continue
                    point = clean_uv[view, frame, ball_index]
                    noisy = np.clip(
                        point + rng.normal(0.0, self.uv_noise_std, size=2), 0.0, 1.0
                    ).astype(np.float32)
                    candidates.append(
                        (noisy, float(rng.uniform(0.75, 1.0)), True, ball_index)
                    )
                    if rng.random() < self.duplicate_probability:
                        duplicate = np.clip(
                            point + rng.normal(0.0, 2.0 * self.uv_noise_std, size=2),
                            0.0,
                            1.0,
                        ).astype(np.float32)
                        candidates.append(
                            (duplicate, float(rng.uniform(0.45, 0.8)), True, ball_index)
                        )
                if rng.random() < self.false_positive_probability:
                    candidates.append(
                        (
                            rng.uniform(0.0, 1.0, size=2).astype(np.float32),
                            float(rng.uniform(0.1, 0.65)),
                            True,
                            -1,
                        )
                    )
                if rng.random() < self.false_positive_probability:
                    coherent_fp[view] = np.clip(
                        coherent_fp[view] + coherent_velocity[view], 0.0, 1.0
                    )
                    candidates.append(
                        (
                            coherent_fp[view].copy(),
                            float(rng.uniform(0.2, 0.7)),
                            True,
                            -1,
                        )
                    )
                rng.shuffle(candidates)
                for detection, item in enumerate(candidates[: self.max_candidates]):
                    uv[view, frame, detection] = item[0]
                    score[view, frame, detection] = item[1]
                    candidate_mask[view, frame, detection] = True
                    visible[view, frame, detection] = item[2]
                    candidate_gt_index[view, frame, detection] = item[3]

        court_kp: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.num_court_keypoints, 2),
            dtype=np.float32,
        )
        base_court = np.stack(
            [
                np.linspace(0.1, 0.9, self.num_court_keypoints, dtype=np.float32),
                np.tile(
                    np.array([0.2, 0.8], dtype=np.float32),
                    self.num_court_keypoints // 2 + 1,
                )[: self.num_court_keypoints],
            ],
            axis=-1,
        )
        court_kp[:valid_views, :valid_frames] = base_court
        court_vis: BoolArray = np.zeros(court_kp.shape[:-1], dtype=np.bool_)
        court_vis[:valid_views, :valid_frames] = True
        frame_mask = np.arange(self.max_frames) < valid_frames
        view_mask = np.arange(self.max_views) < valid_views
        target_ball_mask = np.arange(self.max_balls) < num_balls

        return {
            "scene_format_version": torch.tensor(2, dtype=torch.int64),
            "ball_uv": torch.from_numpy(uv),
            "ball_score": torch.from_numpy(score),
            "ball_candidate_mask": torch.from_numpy(candidate_mask),
            "ball_visible": torch.from_numpy(visible),
            "court_kp": torch.from_numpy(court_kp),
            "court_vis": torch.from_numpy(court_vis),
            "frame_mask": torch.from_numpy(frame_mask),
            "view_mask": torch.from_numpy(view_mask),
            "position_3d": torch.from_numpy(position),
            "ball_present": torch.from_numpy(present),
            "target_ball_mask": torch.from_numpy(target_ball_mask),
            "ball_uv_gt": torch.from_numpy(clean_uv),
            "ball_visible_gt": torch.from_numpy(clean_visible),
            "candidate_gt_index": torch.from_numpy(candidate_gt_index),
        }
