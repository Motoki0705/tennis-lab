"""Lightning DataModule for SLCS training on the issue #634 dataset."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader

from src.tasks.base.data.loader_state import LoaderRNGDataModule
from src.tasks.slcs.configuration import SLCSDataRuntimeConfig
from src.tasks.slcs.data.dataset import SLCSWindowDataset, collate_slcs
from src.tasks.slcs.data.sampling import DomainBalancedSampler


class SLCSDataModule(LoaderRNGDataModule):
    """Builds train/val/test :class:`SLCSWindowDataset` loaders.

    Required ``config.data`` keys: ``dataset_root``, ``split_file``,
    ``batch_size`` and the :class:`SLCSDataConfig` fields (``window_size``,
    ``dino`` section, ...).
    """

    def __init__(
        self, config: SLCSDataRuntimeConfig, *, seed: int | None = None
    ) -> None:
        super().__init__(loader_seed=torch.initial_seed() if seed is None else seed)
        self.config = config
        self.seed = seed
        self.dataset_root = config.dataset_root
        self.split_file = config.split_file
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.overfit = config.overfit
        self.data_config = config.pipeline

        self.train_dataset: SLCSWindowDataset | None = None
        self.val_dataset: SLCSWindowDataset | None = None
        self.test_dataset: SLCSWindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = self._build_dataset("train")
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val")
        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = self._build_dataset("val")
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = self._build_dataset("test")

    def _build_dataset(self, split: str) -> SLCSWindowDataset:
        source_split = "train" if self.overfit else split
        return SLCSWindowDataset(
            dataset_root=self.dataset_root,
            split_file=self.split_file,
            split=source_split,
            augment=split == "train",
            config=self.data_config,
            stride=(
                self.data_config.train_stride
                if source_split == "train"
                else self.data_config.eval_stride
            ),
        )

    def _loader(
        self, dataset: SLCSWindowDataset, *, shuffle: bool, stage: str
    ) -> DataLoader[Any]:
        sampler = None
        sampling = self.config.domain_sampling
        if shuffle and sampling is not None and sampling.enabled:
            if (self.trainer is not None and self.trainer.world_size > 1) or (
                distributed.is_initialized() and distributed.get_world_size() > 1
            ):
                raise RuntimeError(
                    "Domain-balanced sampling does not support distributed training."
                )
            if self.seed is None:
                raise ValueError("Domain-balanced sampling requires run.seed.")
            sampler = DomainBalancedSampler(
                [meta.video_id for meta in dataset.metas],
                sampling.video_domains,
                seed=self.seed,
            )
            if self.trainer is not None:
                sampler.set_epoch(self.trainer.current_epoch)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_slcs,
            drop_last=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            generator=self.loader_generator(stage),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader().")
        return self._loader(self.train_dataset, shuffle=True, stage="train")

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("setup('fit') must run before val_dataloader().")
        return self._loader(self.val_dataset, shuffle=False, stage="val")

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            raise RuntimeError("setup('test') must run before test_dataloader().")
        return self._loader(self.test_dataset, shuffle=False, stage="test")


__all__ = ["SLCSDataModule"]
