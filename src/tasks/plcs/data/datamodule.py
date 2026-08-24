"""Unified PyTorch Lightning DataModule for PLCS."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.base.generate_dataset import CAMERA_VIEW_V2_SELECTOR
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch


class PLCSDataModule(SceneDirectoryDataModule):
    """Lightning DataModule for unified PLCS frame/sequence/multiview training."""

    def __init__(self, config: object) -> None:
        self.plcs_runtime = PLCSTrainingConfig.from_config(config)
        super().__init__(config)

    def _build_collate_fn(self) -> Callable[..., Any] | None:
        return collate_plcs_batch

    def _build_dataset(
        self,
        scene_dir: Path,
        split_file: str,
        augment: bool,
        seed: int | None = None,
    ) -> Dataset:
        return SceneDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.plcs_runtime.raw,
            seed=(
                self._dataset_seed(scene_dir, split_file) if seed is None else seed
            ),
            augment=augment,
            reference_camera_id=(
                self.plcs_runtime.data.evaluation_reference_camera_id
                if (
                    not augment
                    and self.plcs_runtime.court_keypoint_contract.selector
                    == CAMERA_VIEW_V2_SELECTOR
                )
                else None
            ),
        )

    def _dataset_name(self) -> str:
        return "plcs"
