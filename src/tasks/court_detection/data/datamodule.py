"""Mixed-source Court data loading with fixed within-batch source ratios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import Any, cast

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.tasks.base.configuration import (
    as_config_mapping,
)
from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtSourceSplit,
)
from src.tasks.court_detection.data.dataset import CourtDetectionDataset
from src.tasks.court_detection.data.mixed import (
    _SOURCE_ORDER,
    CourtMixedDataConfig,
    MixedSourceBatchSampler,
    _compatible_bundle,
    mixed_court_detection_collate,
)
from src.tasks.court_detection.data.processing.factory import (
    build_court_processing_pipeline,
)
from src.tasks.court_detection.data.processing.pipeline import CourtProcessingPipeline


class CourtDetectionDataModule(pl.LightningDataModule):
    """Build source-specific pipelines and mix them within every train batch."""

    def __init__(
        self,
        config: object,
        *,
        mixed_config: CourtMixedDataConfig | None = None,
    ) -> None:
        super().__init__()
        runtime = CourtTrainingConfig.from_config(config)
        self.data_config = runtime.data
        if mixed_config is None:
            values = as_config_mapping(config, path="configuration")
            mixed_config = CourtMixedDataConfig.from_mapping(
                values.get("mixed"), runtime=runtime
            )
        self.mixed_config = mixed_config
        self.batch_size = runtime.data.batch_size
        self.num_workers = runtime.data.num_workers
        self.pin_memory = runtime.data.pin_memory
        self.seed = runtime.shared.run.seed
        self.pose_variant = runtime.loss.pose.enabled

        self._train_pipelines: dict[str, CourtProcessingPipeline] = {}
        self._eval_pipelines: dict[str, CourtProcessingPipeline] = {}
        for name in _SOURCE_ORDER:
            source = mixed_config.sources[name]
            source_data_config = replace(runtime.data, source=source)
            require_pose = self.pose_variant and name == "synthetic_court"
            self._train_pipelines[name] = build_court_processing_pipeline(
                source_data_config,
                is_train=True,
                require_pose=require_pose,
            )
            self._eval_pipelines[name] = build_court_processing_pipeline(
                source_data_config,
                is_train=False,
                require_pose=require_pose,
            )

        canonical = self._train_pipelines["synthetic_court"].target_bundle_spec
        for pipeline in (
            *self._train_pipelines.values(),
            *self._eval_pipelines.values(),
        ):
            if not _compatible_bundle(canonical, pipeline.target_bundle_spec):
                raise ValueError(
                    "Mixed Court sources expose incompatible target/head contracts: "
                    f"canonical={canonical!r}, candidate={pipeline.target_bundle_spec!r}."
                )
        if "kp" in canonical.kinds:
            flip_permutations = {
                tuple(pipeline.input_layer.spec.keypoint_flip_permutation)
                for pipeline in self._train_pipelines.values()
            }
            if len(flip_permutations) != 1:
                raise ValueError(
                    "Mixed Court KP sources disagree on horizontal-flip identity."
                )
        self.target_bundle_spec = canonical

        # Match the existing pose DataModule boundary: validate every synthetic
        # pose authority before model construction, accelerator setup, or workers.
        if self.pose_variant:
            train_pipeline = self._train_pipelines["synthetic_court"]
            eval_pipeline = self._eval_pipelines["synthetic_court"]
            for split in train_pipeline.input_layer.available_splits:
                pipeline = train_pipeline if split == "train" else eval_pipeline
                pipeline.preflight(pipeline.input_layer.records(split))

        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None
        self._train_source_lengths: Mapping[str, int] | None = None

    @staticmethod
    def _source_dataset(
        *,
        split: CourtSourceSplit,
        pipeline: CourtProcessingPipeline,
    ) -> CourtDetectionDataset | None:
        if split not in pipeline.input_layer.available_splits:
            return None
        return CourtDetectionDataset(
            pipeline.input_layer.records(split),
            pipeline=pipeline,
        )

    def _datasets_for_split(
        self,
        split: CourtSourceSplit,
    ) -> dict[str, CourtDetectionDataset]:
        pipelines = self._train_pipelines if split == "train" else self._eval_pipelines
        datasets: dict[str, CourtDetectionDataset] = {}
        for name in _SOURCE_ORDER:
            dataset = self._source_dataset(split=split, pipeline=pipelines[name])
            if dataset is not None:
                datasets[name] = dataset
        if not datasets:
            raise ValueError(f"Mixed Court sources expose no {split!r} samples.")
        return datasets

    @staticmethod
    def _concat(datasets: Mapping[str, Dataset[Any]]) -> Dataset[Any]:
        return cast(Dataset[Any], ConcatDataset(list(datasets.values())))

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", "validate", "test", None):
            return
        if stage in ("fit", None):
            train = self._datasets_for_split("train")
            if tuple(train) != _SOURCE_ORDER:
                raise ValueError(
                    "Every configured mixed source requires train samples."
                )
            self.train_dataset = self._concat(train)
            self._train_source_lengths = MappingProxyType(
                {name: len(dataset) for name, dataset in train.items()}
            )
        if stage in ("fit", "validate", None):
            self.val_dataset = self._concat(self._datasets_for_split("val"))
        if stage in ("test", None):
            self.test_dataset = self._concat(self._datasets_for_split("test"))

    @staticmethod
    def _require_dataset(
        dataset: Dataset[Any] | None,
        *,
        stage: str,
    ) -> Dataset[Any]:
        if dataset is None:
            raise RuntimeError(
                f"CourtDetectionDataModule.setup({stage!r}) was not called."
            )
        return dataset

    def train_dataloader(self) -> DataLoader[Any]:
        dataset = self._require_dataset(self.train_dataset, stage="fit")
        if self._train_source_lengths is None:
            raise RuntimeError("Mixed Court train source lengths are unresolved.")
        sampler = MixedSourceBatchSampler(
            self._train_source_lengths,
            self.mixed_config.train_batch_counts,
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                mixed_court_detection_collate,
                bundle=self.target_bundle_spec,
                require_pose_supervision=self.pose_variant,
            ),
        )

    def _eval_loader(
        self,
        dataset: Dataset[Any] | None,
        *,
        stage: str,
    ) -> DataLoader[Any]:
        return DataLoader(
            self._require_dataset(dataset, stage=stage),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                mixed_court_detection_collate,
                bundle=self.target_bundle_spec,
                require_pose_supervision=self.pose_variant,
            ),
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return self._eval_loader(self.val_dataset, stage="validate")

    def test_dataloader(self) -> DataLoader[Any]:
        return self._eval_loader(self.test_dataset, stage="test")
