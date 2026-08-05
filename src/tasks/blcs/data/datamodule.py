"""PyTorch Lightning DataModule for BLCS."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_and_adapt_blcs_batch,
)


class BLCSDataModule(SceneDirectoryDataModule):
    """Lightning DataModule for unified BLCS single/multiview training."""

    def _build_collate_fn(self) -> Callable[..., Any] | None:
        config: Any = self.config
        self.input_profile = str(config["model"]["io"]["input_profile"])
        if self.input_profile not in {"single", "multiview"}:
            raise ValueError(
                "Invalid model.io.input_profile="
                f"'{self.input_profile}'. Supported: ['single', 'multiview']"
            )
        return partial(
            collate_and_adapt_blcs_batch,
            input_profile=self.input_profile,
        )

    def _build_dataset(
        self,
        scene_dir: Path,
        split_file: str,
        augment: bool,
    ) -> Dataset:
        return BallTrajectoryDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.config,
            augment=augment,
        )

    def _dataset_name(self) -> str:
        return "blcs"
