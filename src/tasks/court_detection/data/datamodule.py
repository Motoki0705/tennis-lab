"""The sole Lightning DataModule for composable Court detection."""

from __future__ import annotations

from functools import partial
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.collate import court_detection_collate
from src.tasks.court_detection.data.contracts import (
    CourtSourceSplit,
    CourtTargetBundleSpec,
)
from src.tasks.court_detection.data.dataset import CourtDetectionDataset
from src.tasks.court_detection.data.processing.factory import (
    build_court_processing_pipeline,
)
from src.tasks.court_detection.data.processing.pipeline import CourtProcessingPipeline


class CourtDetectionDataModule(pl.LightningDataModule):
    """Resolve source and targets once, then reuse one Dataset implementation."""

    def __init__(self, config: object) -> None:
        super().__init__()
        runtime = CourtTrainingConfig.from_config(config)
        self.data_config = runtime.data
        self.batch_size = runtime.data.batch_size
        self.num_workers = runtime.data.num_workers
        self.pin_memory = runtime.data.pin_memory
        self.pose_variant = bool(getattr(runtime.loss.pose, "enabled", False))

        if self.pose_variant:
            self._train_pipeline = build_court_processing_pipeline(
                self.data_config,
                is_train=True,
                require_pose=True,
            )
            self._eval_pipeline = build_court_processing_pipeline(
                self.data_config,
                is_train=False,
                require_pose=True,
            )
        else:
            self._train_pipeline = build_court_processing_pipeline(
                self.data_config, is_train=True
            )
            self._eval_pipeline = build_court_processing_pipeline(
                self.data_config, is_train=False
            )
        if (
            self._train_pipeline.target_bundle_spec
            != self._eval_pipeline.target_bundle_spec
        ):
            raise ValueError("Court train/eval target bundle contracts disagree.")
        self.target_bundle_spec: CourtTargetBundleSpec = (
            self._train_pipeline.target_bundle_spec
        )

        # Pose authority is scanned synchronously here: Runner constructs this
        # DataModule before the model, Trainer, accelerator selection, or workers.
        if self.pose_variant:
            for split in self._train_pipeline.input_layer.available_splits:
                pipeline = (
                    self._train_pipeline if split == "train" else self._eval_pipeline
                )
                pipeline.preflight(pipeline.input_layer.records(split))

        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

    def _create_dataset(
        self,
        *,
        split: CourtSourceSplit,
        pipeline: CourtProcessingPipeline,
    ) -> CourtDetectionDataset:
        records = pipeline.input_layer.records(split)
        return CourtDetectionDataset(records, pipeline=pipeline)

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", "validate", "test", None):
            return
        if stage in ("fit", None):
            self.train_dataset = self._create_dataset(
                split="train", pipeline=self._train_pipeline
            )
        if stage in ("fit", "validate", None):
            self.val_dataset = self._create_dataset(
                split="val", pipeline=self._eval_pipeline
            )
        if stage in ("test", None):
            self.test_dataset = self._create_dataset(
                split="test", pipeline=self._eval_pipeline
            )

    @staticmethod
    def _require_dataset(
        dataset: Dataset[Any] | None, *, stage: str
    ) -> Dataset[Any]:
        if dataset is None:
            raise RuntimeError(
                f"CourtDetectionDataModule.setup({stage!r}) was not called."
            )
        return dataset

    def _loader(
        self,
        dataset: Dataset[Any] | None,
        *,
        stage: str,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            self._require_dataset(dataset, stage=stage),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                court_detection_collate, bundle=self.target_bundle_spec
            ),
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            stage="fit",
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.val_dataset,
            stage="validate",
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.test_dataset,
            stage="test",
            shuffle=False,
            drop_last=False,
        )


__all__ = ["CourtDetectionDataModule"]
