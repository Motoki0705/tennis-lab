"""Mixed-source Court data loading with fixed within-batch source ratios."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Sampler

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
    CourtTargetBundleSpec,
)
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
            synthetic.schema != "v3" or synthetic.court_scope != "target_court"
        ):
            raise SemanticConfigurationError(
                "Mixed KP training requires Synthetic Court V3 with "
                "court_scope='target_court'."
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
    raw_output = court_detection_collate(dense_only_batch, bundle=bundle)
    if not isinstance(raw_output, dict) or any(
        not isinstance(key, str) for key in raw_output
    ):
        raise TypeError("Court collate must return a string-keyed dictionary.")
    output: dict[str, object] = dict(raw_output)
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


__all__ = [
    "CourtMixedDataConfig",
    "MixedSourceBatchSampler",
    "mixed_court_detection_collate",
]
