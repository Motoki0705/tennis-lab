"""LightningDataModule that wires the DanceTrack dataset and collate."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.scene_model.collate_tracking import SceneBatch, collate_tracking
from src.datasets.scene_model.dancetrack import DancetrackDataset
from src.training.base.datamodule import BaseDataModule


class DancetrackDataModule(BaseDataModule[DancetrackDataset, SceneBatch]):
    """DataModule that provides train/val DataLoaders for DanceTrack."""

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
        elif stage in ("validate", "test"):
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val")

    def _build_dataset(self, split: str) -> DancetrackDataset:
        return DancetrackDataset(self.dataset_cfg, split=split, debug=self.debug_cfg)

    def train_dataloader(self) -> DataLoader[SceneBatch]:
        """Return the training DataLoader configured via YAML."""
        return self._make_loader(self.train_dataset, "train", "train", True)

    def val_dataloader(self) -> DataLoader[SceneBatch]:
        """Return the validation DataLoader configured via YAML."""
        return self._make_loader(self.val_dataset, "val", "val", False)

    def collate_fn(self) -> Callable[..., Any] | None:
        """Return the custom collate function for SceneBatch."""
        return collate_tracking


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
