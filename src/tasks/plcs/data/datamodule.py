"""Unified PyTorch Lightning DataModule for PLCS."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.tasks.plcs.data.dataset import SceneDataset, collate_and_adapt_plcs_batch


class PLCSDataModule(SceneDirectoryDataModule):
    """Lightning DataModule for unified PLCS frame/sequence/multiview training."""

    default_scene_dir: str = "data/plcs"
    default_batch_size: int = 64

    @staticmethod
    def _infer_input_profile_from_model_name(model_name: str) -> str:
        if model_name == "plcs":
            return "frame"
        if model_name in {"plcs_multiview", "plcs_multiview_axial"}:
            return "multiview"
        raise ValueError(
            f"Unknown model.name='{model_name}' for input profile inference."
        )

    def _build_collate_fn(self) -> Callable[..., Any] | None:
        data_cfg = self.config.get("data", {})
        self.adapter_camera_index = int(data_cfg.get("adapter_camera_index", 0))

        model_cfg = self.config.get("model", {})
        io_cfg = model_cfg.get("io", {})
        input_profile = io_cfg.get("input_profile")
        if input_profile is None:
            input_profile = self._infer_input_profile_from_model_name(
                str(model_cfg.get("name", "plcs"))
            )
        self.input_profile = str(input_profile)

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
            config=self.config,
            augment=augment,
        )

    def _dataset_name(self) -> str:
        return "plcs"
