"""LightningDataModule wiring for the Tennis pose system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.tennis import TennisSceneWindowDataset
from src.training.base.datamodule import BaseDataModule


class TennisPoseDataModule(BaseDataModule[TennisSceneWindowDataset, dict[str, Any]]):
    """DataModule that provides DataLoaders for Tennis pose training."""

    def __init__(
        self,
        dataset_cfg: DictConfig | Mapping[str, Any] | None,
        debug_cfg: DictConfig | Mapping[str, Any] | None,
    ) -> None:
        super().__init__(dataset_cfg, debug_cfg)

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
        return self._make_loader(self.train_dataset, "train", "train", True)

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the validation DataLoader configured via dataset_cfg."""
        return self._make_loader(self.val_dataset, "val", "val", False)

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        """Return the test DataLoader configured via dataset_cfg."""
        return self._make_loader(self.test_dataset, "test", "test", False)


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
