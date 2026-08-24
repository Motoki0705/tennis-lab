"""PyTorch Lightning DataModule for BLCS."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.base.generate_dataset import CAMERA_VIEW_V2_SELECTOR
from src.tasks.blcs.configuration import parse_court_keypoint_contract
from src.tasks.blcs.data.dataset import BallTrajectoryDataset


class BLCSDataModuleHooks:
    """Task-local dataset/collate hooks shared by fixed and chunked loaders."""

    config: Any
    _collate_fn: Callable[..., Any]

    def _build_collate_fn(self) -> Callable[..., Any] | None:
        return self._collate_fn

    def _build_dataset(
        self,
        scene_dir: Path,
        split_file: str,
        augment: bool,
    ) -> Dataset:
        contract = parse_court_keypoint_contract(self.config)
        return BallTrajectoryDataset(
            scene_dir=scene_dir,
            split_file=split_file,
            config=self.config,
            augment=augment,
            reference_camera_id=(
                str(self.config.data.evaluation_reference_camera_id)
                if not augment and contract.selector == CAMERA_VIEW_V2_SELECTOR
                else None
            ),
        )

    def _dataset_name(self) -> str:
        return "blcs"


class BLCSDataModule(BLCSDataModuleHooks, SceneDirectoryDataModule):
    """Lightning DataModule for unified BLCS single/multiview training."""

    def __init__(self, config: Any, *, collate_fn: Callable[..., Any]) -> None:
        self._collate_fn = collate_fn
        super().__init__(config)
