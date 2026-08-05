"""Unified PyTorch Lightning DataModule for PLCS."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.dataset import SceneDataset, collate_and_adapt_plcs_batch


class PLCSDataModule(SceneDirectoryDataModule):
    """Lightning DataModule for unified PLCS frame/sequence/multiview training."""

    def __init__(self, config: object) -> None:
        self.plcs_runtime = PLCSTrainingConfig.from_config(config)
        super().__init__(config)

    def _build_collate_fn(self) -> Callable[..., Any] | None:
        self.adapter_camera_index = self.plcs_runtime.data.adapter_camera_index
        input_profile = self.plcs_runtime.data.input_profile
        if input_profile is None:
            raise ValueError("Non-tracking PLCS data requires model.io.input_profile.")
        self.input_profile = input_profile

        return partial(
            collate_and_adapt_plcs_batch,
            input_profile=self.input_profile,
            camera_index=self.adapter_camera_index,
        )

    def _build_dataset(
        self,
        scene_dir: Path,
        split_file: str,
        augment: bool,
    ) -> Dataset:
        return SceneDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.plcs_runtime.raw,
            augment=augment,
        )

    def _dataset_name(self) -> str:
        return "plcs"
