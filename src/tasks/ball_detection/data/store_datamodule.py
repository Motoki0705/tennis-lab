"""Lightning DataModule over the unified ball frame store (``data.source=store``).

* **Sources**: ``data.sources`` lists the store sources used for every split.
* **Train**: every ``data.train_stride``-th window of the train clips. An epoch
  is ``data.train_sampling.windows_per_epoch`` windows drawn with the fixed
  per-source proportions ``data.train_sampling.source_weights``
  (:class:`SourceMixSampler`); consecutive windows overlap, so an epoch is a
  sample of the pool rather than one pass over it.
* **Windows** are ``model.num_frames`` long and must contain a supervised frame.
* **Val**: every ``data.eval_stride``-th window plus real tail backfill,
  independent of labels. Short clips/gaps fail explicitly.
* **Test**: every supervised ``data.eval_stride``-th window, in store order.
* **Supervision**: ``data.supervision`` assigns every point kind a role
  (:mod:`.supervision`); samples carry the per-frame ``supervised`` mask.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler

from src.tasks.ball_detection.configuration import BallRuntimePaths, validate_data
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.components.source_mix_sampler import (
    SourceMixSampler,
)
from src.tasks.ball_detection.data.store import (
    BallFrameStore,
    Source,
    Split,
)
from src.tasks.ball_detection.data.store_dataset import (
    BallStoreDataset,
    select_store_windows,
)
from src.tasks.ball_detection.data.supervision import (
    FrameSupervision,
    FrameSupervisionPolicy,
    resolve_frame_supervision,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallStoreDataModule(pl.LightningDataModule):
    """Mixed-source training and evaluation on one ball frame store version.

    ``data.train_sampling: null`` draws every train window once per epoch in a
    shuffled order (natural source proportions).
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        paths = BallRuntimePaths.from_config(config)
        data_cfg = validate_data(config, paths=paths)
        self.batch_size = int(data_cfg["batch_size"])
        self.num_workers = int(data_cfg["num_workers"])
        self.pin_memory = bool(data_cfg["pin_memory"])
        self.store = BallFrameStore(paths.data(str(data_cfg["data_dir"])))
        self.sources: tuple[Source, ...] = tuple(
            str(name) for name in cast(Sequence[object], data_cfg["sources"])
        )
        missing = sorted(set(self.sources) - {clip.source for clip in self.store.clips})
        if missing:
            raise ValueError(
                f"data.sources names source(s) absent from {self.store.directory}: {missing}"
            )
        self.policy = FrameSupervisionPolicy.from_mapping(
            cast(Mapping[str, Sequence[str]], data_cfg["supervision"])
        )
        self.supervision: FrameSupervision = resolve_frame_supervision(
            self.store, self.policy
        )
        eval_stride = (
            int(config.model.num_frames)
            if data_cfg["eval_stride"] is None
            else int(data_cfg["eval_stride"])
        )
        self.strides: dict[Split, int] = {
            "train": int(data_cfg["train_stride"]),
            "val": eval_stride,
            "test": eval_stride,
        }
        self.augmentation_config = data_cfg["augmentation"]
        sampling = data_cfg["train_sampling"]
        self.train_sampling: Mapping[str, Any] | None = (
            None if sampling is None else cast(Mapping[str, Any], sampling)
        )
        if self.train_sampling is not None:
            weights = cast(Mapping[str, Any], self.train_sampling["source_weights"])
            if set(weights) != set(self.sources):
                raise ValueError(
                    "data.train_sampling.source_weights must name exactly data.sources: "
                    f"{sorted(weights)} vs {sorted(self.sources)}"
                )
        self.selection_stats: dict[str, dict[str, dict[str, int]]] = {}
        self.train_dataset: BallStoreDataset | None = None
        self.val_dataset: BallStoreDataset | None = None
        self.test_dataset: BallStoreDataset | None = None

    def create_dataset(
        self, split: Split, *, augmentation: BallDetectionAugmentation | None
    ) -> BallStoreDataset:
        """Select the supervised windows of one split and wrap them."""
        selection = select_store_windows(
            self.store,
            self.supervision,
            self.store.split_clips(split, self.sources),
            length=int(self.config.model.num_frames),
            stride=self.strides[split],
            validation=split == "val",
        )
        self.selection_stats[split] = selection.stats
        print(
            f"[ball_detection] {split} windows: {json.dumps(selection.stats, sort_keys=True)}"
        )
        if not selection.windows:
            raise RuntimeError(
                f"No supervised {split} windows in {self.store.directory} "
                f"for sources {list(self.sources)}"
            )
        return BallStoreDataset(
            store=self.store,
            supervision=self.supervision,
            windows=selection.windows,
            config=self.config,
            augmentation=augmentation,
        )

    def setup(self, stage: str | None = None) -> None:
        train_augmentation = BallDetectionAugmentation(self.augmentation_config)
        eval_augmentation = BallDetectionAugmentation.from_eval_config(
            self.augmentation_config
        )
        if stage in {"fit", None} and self.train_dataset is None:
            self.train_dataset = self.create_dataset(
                "train", augmentation=train_augmentation
            )
        if stage in {"fit", "validate", None} and self.val_dataset is None:
            self.val_dataset = self.create_dataset(
                "val", augmentation=eval_augmentation
            )
        if stage in {"test", None} and self.test_dataset is None:
            self.test_dataset = self.create_dataset(
                "test", augmentation=eval_augmentation
            )

    def train_sampler(self) -> SourceMixSampler | RandomSampler:
        dataset = self.train_dataset
        if dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader().")
        if self.train_sampling is None:
            return RandomSampler(dataset)
        source_indices: dict[str, list[int]] = {name: [] for name in self.sources}
        for index in range(len(dataset)):
            source_indices[dataset.source_of(index)].append(index)
        weights = cast(Mapping[str, Any], self.train_sampling["source_weights"])
        return SourceMixSampler(
            source_indices=source_indices,
            weights={str(name): float(weight) for name, weight in weights.items()},
            samples_per_epoch=int(self.train_sampling["windows_per_epoch"]),
            seed=int(self.train_sampling["seed"]),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            cast(BallStoreDataset, self.train_dataset),
            batch_size=self.batch_size,
            sampler=self.train_sampler(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def _eval_loader(
        self, dataset: BallStoreDataset | None, stage: str
    ) -> DataLoader[Any]:
        if dataset is None:
            raise RuntimeError(f"setup('{stage}') must run before its dataloader.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return self._eval_loader(self.val_dataset, "validate")

    def test_dataloader(self) -> DataLoader[Any]:
        return self._eval_loader(self.test_dataset, "test")


__all__ = ["BallStoreDataModule"]
