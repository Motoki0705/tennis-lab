"""Manifest-backed Lightning data lifecycle with resumable window sampling."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from src.tasks.ball_refiner_3d.configuration.training import RefinerTrainingSettings
from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.ball_refiner_3d.data.preprocessing import prepare
from src.tasks.ball_refiner_3d.data.sampling import sample_batch
from src.tasks.ball_refiner_3d.data.schema import PreparedRally
from src.tasks.ball_refiner_3d.physics.targets import FlightClock


class WindowBatches(IterableDataset[dict[str, Tensor]]):
    def __init__(self, module: RefinerDataModule, epoch: int) -> None:
        super().__init__()
        self.module = module
        self.epoch = epoch

    def __len__(self) -> int:
        updates = self.module.settings.updates
        return min(
            updates.evaluate_every, updates.steps - self.epoch * updates.evaluate_every
        )

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        module = self.module
        settings = module.settings
        if module._train_epoch != self.epoch:
            module.training_data = prepare(
                module.dataset.split("train"),
                3,
                settings.augmentation,
                settings.runtime.run.seed + 10000 + self.epoch,
                event_sigma_frames=settings.event.sigma_frames,
            )
            module._train_epoch = self.epoch
        for _ in range(len(self)):
            yield sample_batch(
                module.training_data,
                settings.updates.batch_size,
                settings.window,
                module.sampling,
                torch.device("cpu"),
            )


class RefinerDataModule(pl.LightningDataModule):
    """Rally NPZ storage intentionally does not pretend to be SceneDirectory data."""

    def __init__(self, settings: RefinerTrainingSettings) -> None:
        super().__init__()
        self.settings = settings
        self.dataset = SharedDataset(settings.dataset)
        if min(len(r.xyz) for r in self.dataset.rallies) < settings.window.length:
            raise ValueError("Training window exceeds the shortest shared rally")
        self.clock = FlightClock.of([r.physics for r in self.dataset.rallies])
        self.training_data: list[PreparedRally] = []
        self.validation: list[PreparedRally] = []
        self.test: list[PreparedRally] = []
        self._train_epoch = -1
        self.sampling = np.random.default_rng(settings.runtime.run.seed + 300)
        self.loader_generator = torch.Generator().manual_seed(
            settings.runtime.run.seed + 400
        )

    def setup(self, stage: str | None = None) -> None:
        settings = self.settings
        evaluation = replace(
            settings.augmentation,
            event_probability=settings.raw["data"]["evaluation_event_probability"],
        )
        seed = settings.raw["data"]["evaluation_seed"]
        if stage in (None, "fit", "validate") and not self.validation:
            self.validation = prepare(
                self.dataset.split("val"),
                3,
                evaluation,
                seed,
                event_sigma_frames=settings.event.sigma_frames,
            )
        if stage in (None, "test") and not self.test:
            self.training_data.clear()
            self.validation.clear()
            self.test = prepare(
                self.dataset.split("test"),
                3,
                evaluation,
                seed,
                event_sigma_frames=settings.event.sigma_frames,
            )

    def train_dataloader(self) -> DataLoader:
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        return DataLoader(
            WindowBatches(self, epoch),
            batch_size=None,
            num_workers=0,
            pin_memory=self.settings.raw["data"]["pin_memory"],
            generator=self.loader_generator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation,
            batch_size=None,
            num_workers=0,
            generator=self.loader_generator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=None, num_workers=0, generator=self.loader_generator
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampling_rng": self.sampling.bit_generator.state,
            "loader_rng": self.loader_generator.get_state(),
            "manifest_sha256": self.dataset.manifest_hash,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["manifest_sha256"] != self.dataset.manifest_hash:
            raise ValueError("Cannot resume against a different dataset manifest")
        self.sampling.bit_generator.state = state_dict["sampling_rng"]
        self.loader_generator.set_state(state_dict["loader_rng"])
