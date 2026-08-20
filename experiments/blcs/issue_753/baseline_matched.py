"""Baseline-matched synthetic data and runner for the Issue #753 model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.blcs.configuration import TrackQueryModelConfig, parse_model_config
from src.tasks.blcs.model_io.factory import (
    TrackQueryBoundModelIO,
    compose_blcs_model_io,
)
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)

FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int64]


class BaselineMatchedSyntheticDataset(Dataset[dict[str, Tensor]]):
    """Adapt the #648 deterministic synthetic recipe to fixed lifecycle slots."""

    def __init__(self, config: Any, *, split: str) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test, got {split!r}")
        data = config.data
        self.split = split
        self.num_samples = int(data.split_sizes[split])
        self.seed = int(data.seed) + {
            "train": 0,
            "val": 100_000,
            "test": 200_000,
        }[split]
        self.max_frames = int(data.max_frames)
        self.min_frames = int(data.min_frames)
        self.max_views = int(data.max_views)
        self.min_views = int(data.min_views)
        self.max_balls = int(data.max_balls)
        self.min_balls = int(data.min_balls)
        self.num_queries = int(config.model.num_queries)
        self.dropout_probability = float(data.dropout_probability)
        self.false_positive_probability = float(data.false_positive_probability)
        self.uv_noise_std = float(data.uv_noise_std)
        self.num_court_keypoints = int(data.num_court_keypoints)
        if not 0 < self.min_frames <= self.max_frames:
            raise ValueError("data frame bounds must be positive and ordered")
        if not 0 < self.min_views <= self.max_views:
            raise ValueError("data view bounds must be positive and ordered")
        if not 0 < self.min_balls <= self.max_balls <= self.num_queries:
            raise ValueError(
                "data ball bounds must be positive and fit model.num_queries"
            )
        for name, probability in (
            ("dropout_probability", self.dropout_probability),
            ("false_positive_probability", self.false_positive_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"data.{name} must be in [0, 1]")
        if self.uv_noise_std < 0.0:
            raise ValueError("data.uv_noise_std must be non-negative")
        self.scenes = [
            Path(f"{split}_{index:06d}.synthetic")
            for index in range(self.num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _trajectory(rng: np.random.Generator, length: int) -> FloatArray:
        time = np.linspace(0.0, 1.0, length, dtype=np.float32)
        start = np.asarray(
            rng.uniform([0.15, 0.15, 0.08], [0.85, 0.85, 0.35]),
            dtype=np.float32,
        )
        velocity = np.asarray(
            rng.uniform([-0.35, -0.45, 0.20], [0.35, 0.45, 0.65]),
            dtype=np.float32,
        )
        position = start[None] + time[:, None] * velocity[None]
        position[:, 2] += -0.55 * time * time
        position[:, 2] = np.abs(position[:, 2])
        return cast(
            "FloatArray",
            np.asarray(np.clip(position, 0.0, 1.0), dtype=np.float32),
        )

    @staticmethod
    def _project(position: FloatArray, camera_index: int) -> FloatArray:
        angle = 2.0 * np.pi * camera_index / 8.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = (position[..., 0] - 0.5) * cos_a - (
            position[..., 1] - 0.5
        ) * sin_a
        depth = (
            1.8
            + (position[..., 0] - 0.5) * sin_a
            + (position[..., 1] - 0.5) * cos_a
        )
        u = 0.5 + 0.65 * x / depth
        v = 0.82 - 0.65 * position[..., 2] / depth
        return cast(
            "FloatArray",
            np.asarray(np.stack([u, v], axis=-1), dtype=np.float32),
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(self.seed + index)
        valid_frames = int(rng.integers(self.min_frames, self.max_frames + 1))
        valid_views = int(rng.integers(self.min_views, self.max_views + 1))
        num_balls = int(rng.integers(self.min_balls, self.max_balls + 1))
        target_order = np.arange(self.num_queries)
        observation_order = np.arange(self.num_queries)
        if self.split == "train":
            target_order = rng.permutation(target_order)
            observation_order = rng.permutation(observation_order)

        physical_position: FloatArray = np.zeros(
            (self.max_frames, self.max_balls, 3), dtype=np.float32
        )
        physical_presence: BoolArray = np.zeros(
            (self.max_frames, self.max_balls), dtype=np.bool_
        )
        physical_uv: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.max_balls, 2),
            dtype=np.float32,
        )
        physical_visible: BoolArray = np.zeros(
            (self.max_views, self.max_frames, self.max_balls), dtype=np.bool_
        )
        for ball_index in range(num_balls):
            start = int(rng.integers(0, max(valid_frames // 3, 1)))
            end = int(
                rng.integers(max(start + 2, valid_frames * 2 // 3), valid_frames + 1)
            )
            trajectory = self._trajectory(rng, end - start)
            physical_position[start:end, ball_index] = trajectory
            physical_presence[start:end, ball_index] = True
            for view in range(valid_views):
                projected = self._project(trajectory, view)
                visible = np.logical_and(
                    (projected >= 0.0).all(-1), (projected <= 1.0).all(-1)
                )
                physical_uv[view, start:end, ball_index] = projected
                physical_visible[view, start:end, ball_index] = visible

        target_position: FloatArray = np.zeros(
            (self.max_frames, self.num_queries, 3), dtype=np.float32
        )
        target_presence: BoolArray = np.zeros(
            (self.max_frames, self.num_queries), dtype=np.bool_
        )
        target_instance_id: IntArray = np.full(
            (self.max_frames, self.num_queries), -1, dtype=np.int64
        )
        ball_uv: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.num_queries, 2),
            dtype=np.float32,
        )
        ball_visible: BoolArray = np.zeros(
            (self.max_views, self.max_frames, self.num_queries), dtype=np.bool_
        )
        candidate_mask: BoolArray = np.zeros_like(ball_visible)
        for ball_index in range(num_balls):
            target_slot = int(target_order[ball_index])
            observation_slot = int(observation_order[ball_index])
            present = physical_presence[:, ball_index]
            target_position[:, target_slot] = physical_position[:, ball_index]
            target_presence[:, target_slot] = present
            target_instance_id[present, target_slot] = ball_index
            candidate_mask[:valid_views, :, observation_slot] = present[None]
            for view in range(valid_views):
                for frame in range(valid_frames):
                    if not present[frame]:
                        continue
                    point_is_visible = bool(physical_visible[view, frame, ball_index])
                    dropped = point_is_visible and (
                        rng.random() < self.dropout_probability
                    )
                    if point_is_visible and not dropped:
                        point = physical_uv[view, frame, ball_index]
                        ball_uv[view, frame, observation_slot] = np.clip(
                            point + rng.normal(0.0, self.uv_noise_std, size=2),
                            0.0,
                            1.0,
                        ).astype(np.float32)
                        ball_visible[view, frame, observation_slot] = True
                    elif rng.random() < self.false_positive_probability:
                        ball_uv[view, frame, observation_slot] = rng.uniform(
                            0.0, 1.0, size=2
                        ).astype(np.float32)
                        ball_visible[view, frame, observation_slot] = True

        target_velocity = np.zeros_like(target_position)
        consecutive = target_presence[1:] & target_presence[:-1]
        delta = target_position[1:] - target_position[:-1]
        target_velocity[1:] = np.where(consecutive[..., None], delta, 0.0)

        court_kp: FloatArray = np.zeros(
            (
                self.max_views,
                self.max_frames,
                self.num_court_keypoints,
                2,
            ),
            dtype=np.float32,
        )
        base_court = np.stack(
            [
                np.linspace(
                    0.1, 0.9, self.num_court_keypoints, dtype=np.float32
                ),
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

        return {
            "scene_format_version": torch.tensor(3, dtype=torch.int64),
            "ball_uv": torch.from_numpy(ball_uv),
            "ball_visible": torch.from_numpy(ball_visible),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "court_kp": torch.from_numpy(court_kp),
            "court_vis": torch.from_numpy(court_vis),
            "frame_mask": torch.from_numpy(frame_mask),
            "view_mask": torch.from_numpy(view_mask),
            "target_position": torch.from_numpy(target_position),
            "target_velocity": torch.from_numpy(target_velocity),
            "target_presence": torch.from_numpy(target_presence),
            "target_instance_id": torch.from_numpy(target_instance_id),
            "target_slot_mask": torch.from_numpy(target_presence.any(axis=0)),
        }


class BaselineMatchedDataModule(pl.LightningDataModule):
    """Build the deterministic #648-sized synthetic splits."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.batch_size = int(config.data.batch_size)
        self.num_workers = int(config.data.num_workers)
        self.pin_memory = bool(config.data.pin_memory)
        self.train_dataset: BaselineMatchedSyntheticDataset | None = None
        self.val_dataset: BaselineMatchedSyntheticDataset | None = None
        self.test_dataset: BaselineMatchedSyntheticDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_dataset = BaselineMatchedSyntheticDataset(
                self.config, split="train"
            )
            self.val_dataset = BaselineMatchedSyntheticDataset(
                self.config, split="val"
            )
        if stage in {None, "test"}:
            self.test_dataset = BaselineMatchedSyntheticDataset(
                self.config, split="test"
            )

    def _loader(
        self,
        dataset: BaselineMatchedSyntheticDataset | None,
        *,
        shuffle: bool,
    ) -> DataLoader[dict[str, Tensor]]:
        if dataset is None:
            raise RuntimeError("setup() must run before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self.test_dataset, shuffle=False)


class BaselineMatchedTrainingRunner(BaseTrainingRunner):
    """Train current BLCS tracking components on the matched synthetic splits."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return BaselineMatchedDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        del datamodule, steps_per_epoch
        binding = compose_blcs_model_io(config)
        if not isinstance(parse_model_config(config), TrackQueryModelConfig):
            raise TypeError("Issue #753 experiment requires a track-query model")
        return BLCSTrackingLightningModule(
            config,
            model_io=cast("TrackQueryBoundModelIO", binding),
        )

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None,
    ) -> int | None:
        del train_loader
        if not isinstance(datamodule, BaselineMatchedDataModule):
            raise TypeError("unexpected datamodule for baseline-matched experiment")
        train_size = int(config.data.split_sizes.train)
        return math.ceil(train_size / datamodule.batch_size)


__all__ = [
    "BaselineMatchedDataModule",
    "BaselineMatchedSyntheticDataset",
    "BaselineMatchedTrainingRunner",
]
