"""Strict configuration of the ball frame-store build (``configs/generate_dataset.yaml``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from src.tasks.ball_detection.data.store import SPLIT_CODES, Split
from src.tasks.ball_detection.generate_dataset.frame_store.sources import (
    ChatAnnotationSourceConfig,
    MeijiSourceConfig,
    TrackNetSourceConfig,
)
from src.tasks.base.configuration import (
    ConfigMapping,
    as_config_mapping,
    exact_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.utils.configuration import (
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
)
from src.utils.data.splits import GroupSplitConfig
from src.utils.paths import PROJECT_ROOT

ANNOTATION_STATUSES = frozenset({"completed", "partial"})


@dataclass(frozen=True, slots=True)
class ExplicitGroupSplit:
    """Every group of the source is named exactly once (``group -> split``)."""

    groups: Mapping[str, Split]


SourceSplit = ExplicitGroupSplit | GroupSplitConfig


@dataclass(frozen=True, slots=True)
class FrameStoreBuildConfig:
    dataset_dir: Path
    version: str
    jpeg_quality: int
    max_height: int
    workers: int
    tracknet: tuple[TrackNetSourceConfig, ExplicitGroupSplit] | None
    meiji: tuple[MeijiSourceConfig, ExplicitGroupSplit] | None
    chat_annotation: tuple[ChatAnnotationSourceConfig, GroupSplitConfig] | None

    @classmethod
    def from_config(cls, value: object) -> FrameStoreBuildConfig:
        config = exact_config_mapping(
            as_config_mapping(value, path="configuration"),
            path="configuration",
            required_keys={"paths", "sources", "dataset", "workers", "run"},
            optional_keys={"hydra"},
        )
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(
                require_config_mapping(config, "paths", path="configuration"),
                repository_root=PROJECT_ROOT,
            )
        )
        sources = exact_config_mapping(
            require_config_mapping(config, "sources", path="configuration"),
            path="sources",
            required_keys=set(),
            optional_keys={"tracknet", "meiji", "chat_annotation"},
        )
        if not sources:
            raise SemanticConfigurationError("sources must enable at least one ball source.")
        dataset = _mapping(config, "dataset", "configuration", {"version", "jpeg_quality", "max_height"})
        run = _mapping(config, "run", "configuration", {"output_dir"})
        tracknet = meiji = chat = None
        if "tracknet" in sources:
            node = _mapping(sources, "tracknet", "sources", {"root", "fps", "split"})
            fps = _int(node, "fps", "sources.tracknet", minimum=1)
            tracknet = (
                TrackNetSourceConfig(resolver.resolve(PathRole.DATA, _text(node, "root", "sources.tracknet")), Fraction(fps)),
                _explicit_split(node, "sources.tracknet"),
            )
        if "meiji" in sources:
            node = _mapping(sources, "meiji", "sources", {"root", "split"})
            meiji = (
                MeijiSourceConfig(resolver.resolve(PathRole.DATA, _text(node, "root", "sources.meiji"))),
                _explicit_split(node, "sources.meiji"),
            )
        if "chat_annotation" in sources:
            node = _mapping(sources, "chat_annotation", "sources", {"annotation_root", "allowed_statuses", "split"})
            statuses = frozenset(_str_list(node, "allowed_statuses", "sources.chat_annotation"))
            if not statuses or not statuses <= ANNOTATION_STATUSES:
                raise SemanticConfigurationError(
                    f"sources.chat_annotation.allowed_statuses must be a non-empty subset of {sorted(ANNOTATION_STATUSES)}."
                )
            split = _mapping(node, "split", "sources.chat_annotation", {"val_ratio", "test_ratio", "seed"})
            path = "sources.chat_annotation.split"
            val_ratio = _float(split, "val_ratio", path)
            test_ratio = _float(split, "test_ratio", path)
            if val_ratio + test_ratio >= 1.0:
                raise SemanticConfigurationError(f"{path} ratios must sum to less than 1.")
            chat = (
                ChatAnnotationSourceConfig(
                    resolver.resolve(PathRole.OUTPUT, _text(node, "annotation_root", "sources.chat_annotation")),
                    statuses,
                ),
                GroupSplitConfig(val_ratio, test_ratio, _int(split, "seed", path, minimum=0)),
            )
        quality = _int(dataset, "jpeg_quality", "dataset", minimum=50)
        if quality > 100:
            raise SemanticConfigurationError("dataset.jpeg_quality must be <= 100.")
        return cls(
            dataset_dir=resolver.resolve(PathRole.DATA, _text(run, "output_dir", "run")),
            version=_text(dataset, "version", "dataset"),
            jpeg_quality=quality,
            max_height=_int(dataset, "max_height", "dataset", minimum=16),
            workers=_int(config, "workers", "configuration", minimum=1),
            tracknet=tracknet,
            meiji=meiji,
            chat_annotation=chat,
        )


def validate_generate_boundary(config: DictConfig) -> None:
    FrameStoreBuildConfig.from_config(config)


def _mapping(parent: ConfigMapping, key: str, path: str, keys: set[str]) -> ConfigMapping:
    return exact_config_mapping(
        require_config_mapping(parent, key, path=path), path=f"{path}.{key}", required_keys=keys
    )


def _int(mapping: ConfigMapping, key: str, path: str, *, minimum: int) -> int:
    value = cast(int, require_config_value(mapping, key, int, path=path))
    if value < minimum:
        raise SemanticConfigurationError(f"{path}.{key} must be >= {minimum}, got {value}.")
    return value


def _float(mapping: ConfigMapping, key: str, path: str) -> float:
    value = float(cast(float, require_config_value(mapping, key, (float, int), path=path)))
    if not 0.0 <= value < 1.0:
        raise SemanticConfigurationError(f"{path}.{key} must be in [0, 1), got {value}.")
    return value


def _text(mapping: ConfigMapping, key: str, path: str) -> str:
    value = cast(str, require_config_value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise SemanticConfigurationError(f"{path}.{key} must be non-empty and trimmed.")
    return value


def _str_list(mapping: ConfigMapping, key: str, path: str) -> tuple[str, ...]:
    values = cast(list[object], require_config_value(mapping, key, list, path=path))
    if any(type(value) is not str or not value for value in values):
        raise SemanticConfigurationError(f"{path}.{key} must be a list of non-empty strings.")
    return tuple(cast(list[str], values))


def _explicit_split(node: ConfigMapping, path: str) -> ExplicitGroupSplit:
    split = exact_config_mapping(
        require_config_mapping(node, "split", path=path),
        path=f"{path}.split",
        required_keys=set(SPLIT_CODES),
    )
    groups: dict[str, Split] = {}
    for name in SPLIT_CODES:
        for group in _str_list(split, name, f"{path}.split"):
            if group in groups:
                raise SemanticConfigurationError(f"{path}.split lists group {group!r} twice.")
            groups[group] = cast(Split, name)
    if "train" not in groups.values():
        raise SemanticConfigurationError(f"{path}.split must assign at least one train group.")
    return ExplicitGroupSplit(groups)
