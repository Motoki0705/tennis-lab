"""LightningDataModule wiring for the Tennis pose system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datasets.tennis import TennisSceneWindowDataset


class TennisPoseDataModule(LightningDataModule):
    """DataModule that provides DataLoaders for Tennis pose training."""

    def __init__(
        self,
        dataset_cfg: DictConfig | Mapping[str, Any] | None,
        debug_cfg: DictConfig | Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = _to_dict(dataset_cfg)
        self.debug_cfg = _to_dict(debug_cfg)
        self.train_dataset: TennisSceneWindowDataset | None = None
        self.val_dataset: TennisSceneWindowDataset | None = None
        self.test_dataset: TennisSceneWindowDataset | None = None

        seed = self.debug_cfg.get("seed") or self.dataset_cfg.get("seed")
        self._generator = torch.Generator()
        if seed is None:
            self._generator = None
        else:
            self._generator.manual_seed(int(seed))

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets for the requested Lightning stage."""
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")
        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val")
        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self._build_dataset("test")

    def _build_dataset(self, split: str) -> TennisSceneWindowDataset:
        root = self.dataset_cfg.get("root", "data/tennis_autogen")
        name = self.dataset_cfg.get("name") or self.dataset_cfg.get("dataset_name")
        if not name:
            msg = (
                "dataset_cfg.name (or dataset_cfg.dataset_name) must be set for "
                "tennis_multi_cam_3d_pose"
            )
            raise ValueError(msg)
        window_T = int(self.dataset_cfg.get("window_T", 10))
        max_cameras = int(self.dataset_cfg.get("max_cameras", 4))
        max_players = int(self.dataset_cfg.get("max_players", 20))
        num_joints = int(self.dataset_cfg.get("num_joints", 20))
        use_memmap = bool(self.dataset_cfg.get("use_memmap", False))
        min_cameras_val = self.dataset_cfg.get("min_cameras")
        min_cameras = int(min_cameras_val) if min_cameras_val is not None else None
        augment_2d = bool(self.dataset_cfg.get("augment_2d", False))
        return TennisSceneWindowDataset(
            dataset_root=root,
            dataset_name=name,
            split=split,
            window_T=window_T,
            max_cameras=max_cameras,
            max_players=max_players,
            num_joints=num_joints,
            use_memmap=use_memmap,
            min_cameras=min_cameras,
            augment_2d=augment_2d,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the training DataLoader configured via dataset_cfg."""
        dataset = self._require_dataset(self.train_dataset, "train")
        loader_cfg = self.dataset_cfg.get("loader", {}).get("train", {})
        return self._build_loader(
            dataset, loader_cfg, shuffle=loader_cfg.get("shuffle", True)
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the validation DataLoader configured via dataset_cfg."""
        dataset = self._require_dataset(self.val_dataset, "val")
        loader_cfg = self.dataset_cfg.get("loader", {}).get("val", {})
        return self._build_loader(
            dataset, loader_cfg, shuffle=loader_cfg.get("shuffle", False)
        )

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the test DataLoader configured via dataset_cfg."""
        dataset = self._require_dataset(self.test_dataset, "test")
        loader_cfg = self.dataset_cfg.get("loader", {}).get("test", {})
        return self._build_loader(
            dataset, loader_cfg, shuffle=loader_cfg.get("shuffle", False)
        )

    def _build_loader(
        self,
        dataset: TennisSceneWindowDataset,
        loader_cfg: Mapping[str, Any],
        shuffle: bool,
    ) -> DataLoader[dict[str, Any]]:
        batch_size = int(loader_cfg.get("batch_size", 1))
        num_workers = int(loader_cfg.get("num_workers", 0))
        persistent_workers = (
            bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
        )
        generator = self._generator if shuffle and self._generator is not None else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(loader_cfg.get("pin_memory", False)),
            drop_last=bool(loader_cfg.get("drop_last", False)),
            persistent_workers=persistent_workers,
            generator=generator,
        )

    @staticmethod
    def _require_dataset(
        dataset: TennisSceneWindowDataset | None,
        split: str,
    ) -> TennisSceneWindowDataset:
        if dataset is None:
            msg = f"Dataset for split '{split}' has not been set up yet"
            raise RuntimeError(msg)
        return dataset


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
