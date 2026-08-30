"""Mixed-source Court data loading with fixed within-batch source ratios."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import Any, cast

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.court_detection.configuration import (
    CourtSourceConfig,
    CourtTrainingConfig,
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
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
from src.utils.configuration import (
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

_SOURCE_ORDER = ("synthetic_court", "tennis_court_detector")
_MIXED_KP_TARGET_SCHEMAS = frozenset(
    {
        "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1",
        "tennis_court_detector_kp14:gaussian_max_v1",
    }
)
_POSE_FIELDS = {
    "translation_m",
    "rotation",
    "log_focal",
    "intrinsics",
    "semantic_to_physical",
    "raw_pose10d",
}


def _exact(mapping: Mapping[str, object], keys: set[str], *, path: str) -> None:
    unknown = sorted(set(mapping) - keys)
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): "
            f"{', '.join(f'{path}.{key}' for key in unknown)}."
        )
    missing = sorted(keys - set(mapping))
    if missing:
        raise SemanticConfigurationError(
            f"Missing required configuration key(s): "
            f"{', '.join(f'{path}.{key}' for key in missing)}."
        )


@dataclass(frozen=True, slots=True)
class CourtMixedDataConfig:
    """Resolved two-source composition and per-batch sample counts."""

    sources: Mapping[str, CourtSourceConfig]
    train_batch_counts: Mapping[str, int]

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        runtime: CourtTrainingConfig,
    ) -> CourtMixedDataConfig:
        mapping = as_config_mapping(value, path="mixed")
        _exact(mapping, {"sources", "train_batch_counts"}, path="mixed")
        source_mapping = require_config_mapping(mapping, "sources", path="mixed")
        if set(source_mapping) != set(_SOURCE_ORDER):
            raise SemanticConfigurationError(
                "mixed.sources must contain exactly synthetic_court and "
                "tennis_court_detector."
            )

        sources: dict[str, CourtSourceConfig] = {}
        for name in _SOURCE_ORDER:
            raw_source = require_config_mapping(
                source_mapping,
                name,
                path="mixed.sources",
            )
            kind = cast(
                str,
                require_config_value(
                    raw_source,
                    "kind",
                    str,
                    path=f"mixed.sources.{name}",
                ),
            )
            if kind != name:
                raise SemanticConfigurationError(
                    f"mixed.sources.{name}.kind must be {name!r}."
                )
            source: CourtSourceConfig
            if kind == "synthetic_court":
                source = SyntheticCourtSourceConfig.from_mapping(
                    raw_source,
                    resolver=runtime.shared.resolver,
                )
            else:
                source = TennisCourtDetectorSourceConfig.from_mapping(
                    raw_source,
                    resolver=runtime.shared.resolver,
                )
            sources[name] = source

        counts_mapping = require_config_mapping(
            mapping,
            "train_batch_counts",
            path="mixed",
        )
        if set(counts_mapping) != set(_SOURCE_ORDER):
            raise SemanticConfigurationError(
                "mixed.train_batch_counts must contain exactly synthetic_court "
                "and tennis_court_detector."
            )
        counts: dict[str, int] = {}
        for name in _SOURCE_ORDER:
            value_at_source = require_config_value(
                counts_mapping,
                name,
                int,
                path="mixed.train_batch_counts",
            )
            if type(value_at_source) is not int or value_at_source <= 0:
                raise SemanticConfigurationError(
                    f"mixed.train_batch_counts.{name} must be a positive integer."
                )
            counts[name] = value_at_source
        if sum(counts.values()) != runtime.data.batch_size:
            raise SemanticConfigurationError(
                "The mixed source counts must sum to data.batch_size."
            )

        synthetic = cast(SyntheticCourtSourceConfig, sources["synthetic_court"])
        mixes_keypoints = any(
            target.kind == "kp" for target in runtime.data.processing.targets
        )
        if mixes_keypoints and (
            synthetic.schema != "v3" or synthetic.keypoint_court_scope != "target_court"
        ):
            raise SemanticConfigurationError(
                "Mixed KP training requires Synthetic Court V3 with "
                "keypoint_court_scope='target_court'."
            )
        return cls(
            sources=MappingProxyType(sources),
            train_batch_counts=MappingProxyType(counts),
        )


class MixedSourceBatchSampler(Sampler[list[int]]):
    """Yield full batches with an exact count from every source.

    The longest source/count ratio defines the epoch length. Shorter sources are
    reshuffled and cycled, so every yielded batch preserves the requested mix.
    """

    def __init__(
        self,
        source_lengths: Mapping[str, int],
        batch_counts: Mapping[str, int],
        *,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if not source_lengths or set(source_lengths) != set(batch_counts):
            raise ValueError(
                "Mixed source lengths and batch counts must have identical keys."
            )
        names = tuple(source_lengths)
        if any(source_lengths[name] <= 0 for name in names):
            raise ValueError("Every mixed source dataset must be non-empty.")
        if any(batch_counts[name] <= 0 for name in names):
            raise ValueError("Every mixed source batch count must be positive.")
        self.names = names
        self.source_lengths = MappingProxyType(dict(source_lengths))
        self.batch_counts = MappingProxyType(dict(batch_counts))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        offsets: dict[str, int] = {}
        offset = 0
        for name in names:
            offsets[name] = offset
            offset += source_lengths[name]
        self.offsets = MappingProxyType(offsets)
        self._num_batches = max(
            math.ceil(source_lengths[name] / batch_counts[name]) for name in names
        )
        self._epoch = 0

    def __len__(self) -> int:
        return self._num_batches

    def _order(self, length: int, generator: torch.Generator) -> list[int]:
        if not self.shuffle:
            return list(range(length))
        return cast(list[int], torch.randperm(length, generator=generator).tolist())

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self._epoch
        self._epoch += 1
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        orders = {
            name: self._order(self.source_lengths[name], generator)
            for name in self.names
        }
        cursors = {name: 0 for name in self.names}

        for _ in range(self._num_batches):
            batch: list[int] = []
            for name in self.names:
                remaining = self.batch_counts[name]
                while remaining > 0:
                    order = orders[name]
                    cursor = cursors[name]
                    if cursor == len(order):
                        order = self._order(self.source_lengths[name], generator)
                        orders[name] = order
                        cursor = 0
                    take = min(remaining, len(order) - cursor)
                    offset = self.offsets[name]
                    batch.extend(
                        offset + index for index in order[cursor : cursor + take]
                    )
                    cursors[name] = cursor + take
                    remaining -= take
            if self.shuffle:
                permutation = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[index] for index in permutation]
            yield batch


def mixed_court_detection_collate(
    batch: list[dict[str, object]],
    *,
    bundle: CourtTargetBundleSpec,
    require_pose_supervision: bool,
) -> dict[str, object]:
    """Collate dense targets for all samples and pose targets for synthetic only."""
    if not batch:
        raise ValueError("Mixed Court collate requires a non-empty batch.")
    pose_payloads = [sample.get("pose_target") for sample in batch]
    mask = torch.tensor(
        [payload is not None for payload in pose_payloads],
        dtype=torch.bool,
    )
    for sample, payload in zip(batch, pose_payloads, strict=True):
        metadata = sample.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Mixed Court samples require metadata mappings.")
        source_kind = metadata.get("source_kind")
        if source_kind not in _SOURCE_ORDER:
            raise ValueError("Mixed Court sample has an unknown source_kind.")
        if payload is not None and source_kind != "synthetic_court":
            raise ValueError(
                "Court pose supervision is restricted to synthetic_court samples."
            )
        if require_pose_supervision and source_kind == "synthetic_court":
            if payload is None:
                raise ValueError(
                    "Pose-enabled mixed Court batches require every synthetic_court "
                    "sample to provide pose_target."
                )
        elif payload is not None:
            raise ValueError(
                "Pose-disabled mixed Court batches must not provide pose_target."
            )

    dense_only_batch = [
        {key: value for key, value in sample.items() if key != "pose_target"}
        for sample in batch
    ]
    output = cast(
        dict[str, object],
        court_detection_collate(dense_only_batch, bundle=bundle),
    )
    output["pose_supervision_mask"] = mask

    selected = [payload for payload in pose_payloads if payload is not None]
    if selected:
        if not all(isinstance(payload, Mapping) for payload in selected):
            raise ValueError("Court pose targets must be mappings.")
        typed = [cast(Mapping[str, object], payload) for payload in selected]
        if any(set(payload) != _POSE_FIELDS for payload in typed):
            raise ValueError("Court pose target fields changed before collation.")
        stacked: dict[str, Tensor] = {}
        for field in _POSE_FIELDS:
            values = [payload[field] for payload in typed]
            if not all(isinstance(value, Tensor) for value in values):
                raise ValueError("Court pose target values must be tensors.")
            stacked[field] = torch.stack([cast(Tensor, value) for value in values])
        output["pose_target"] = stacked
    return output


def _compatible_bundle(
    canonical: CourtTargetBundleSpec,
    candidate: CourtTargetBundleSpec,
) -> bool:
    if canonical.kinds != candidate.kinds:
        return False
    for kind in canonical.kinds:
        left = canonical.targets[kind]
        right = candidate.targets[kind]
        if left == right:
            continue
        if (
            kind != "kp"
            or frozenset({left.schema, right.schema}) != _MIXED_KP_TARGET_SCHEMAS
            or left.output_channels != right.output_channels
            or left.channel_names != right.channel_names
            or left.target_dtype != right.target_dtype
            or left.precomputed != right.precomputed
        ):
            return False
    return True


class MixedCourtDetectionDataModule(pl.LightningDataModule):
    """Build source-specific pipelines and mix them within every train batch."""

    def __init__(
        self,
        config: object,
        *,
        mixed_config: CourtMixedDataConfig,
    ) -> None:
        super().__init__()
        runtime = CourtTrainingConfig.from_config(config)
        self.data_config = runtime.data
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
                f"MixedCourtDetectionDataModule.setup({stage!r}) was not called."
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


__all__ = [
    "CourtMixedDataConfig",
    "MixedCourtDetectionDataModule",
    "MixedSourceBatchSampler",
    "mixed_court_detection_collate",
]
