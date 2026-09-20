"""Typed recipes for imported observations and BLCS training preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.tasks.base.configuration import exact_config_mapping
from src.tennis_scene.dataset_pipeline.geometry import TriangulationSettings
from src.tennis_scene.generate_dataset.manifest import validate_id_component
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def _resolver(cfg: DictConfig) -> PathResolver:
    return PathResolver(
        RuntimePathRoots.from_mapping(dict(cfg.paths), repository_root=PROJECT_ROOT)
    )


@dataclass(frozen=True)
class BroadcastImportConfig:
    source: Path
    destination: Path
    dataset_id: str
    video_groups: dict[str, str]
    isolated_jump: float
    neighbor_distance: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> BroadcastImportConfig:
        exact_config_mapping(
            cfg,
            path="broadcast_import",
            required_keys={
                "paths",
                "source",
                "destination",
                "dataset_id",
                "video_groups",
                "isolated_jump",
                "neighbor_distance",
            },
        )
        resolver = _resolver(cfg)
        dataset_id = validate_id_component(cfg.dataset_id, field_name="dataset_id")
        groups = dict(cfg.video_groups)
        if not groups:
            raise ValueError("video_groups cannot be empty")
        for clip, group in groups.items():
            if type(clip) is not str or len(clip.split("/")) != 2:
                raise ValueError("video_groups requires explicit recording/clip IDs")
            for part in clip.split("/"):
                validate_id_component(part, field_name="clip_id")
            validate_id_component(group, field_name="video_group")
        for value in (cfg.isolated_jump, cfg.neighbor_distance):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Ball thresholds must be finite numbers")
        if not 0 < cfg.neighbor_distance < cfg.isolated_jump < 1:
            raise ValueError("Require 0 < neighbor_distance < isolated_jump < 1")
        return cls(
            resolver.resolve(PathRole.DATA, cfg.source),
            resolver.resolve(PathRole.DATA, cfg.destination),
            dataset_id,
            groups,
            float(cfg.isolated_jump),
            float(cfg.neighbor_distance),
        )


def validate_broadcast_import_config(cfg: DictConfig) -> None:
    BroadcastImportConfig.from_config(cfg)


def import_broadcast(runtime: BroadcastImportConfig) -> None:
    from src.tennis_scene.dataset_pipeline.legacy import copy_legacy_broadcast

    copy_legacy_broadcast(
        runtime.source,
        runtime.destination,
        dataset_id=runtime.dataset_id,
        video_groups=runtime.video_groups,
        isolated_jump=runtime.isolated_jump,
        neighbor_distance=runtime.neighbor_distance,
    )


@dataclass(frozen=True)
class BLCSPreparationConfig:
    destination: Path
    synthetic_source: Path
    replay_count: int
    source_recipe: str
    source_overrides: tuple[str, ...]
    paths: DictConfig
    video_splits: dict[str, str]
    geometry: TriangulationSettings

    @classmethod
    def from_config(cls, cfg: DictConfig) -> BLCSPreparationConfig:
        exact_config_mapping(
            cfg,
            path="blcs_preparation",
            required_keys={
                "paths",
                "destination",
                "synthetic_source",
                "replay_count",
                "source_recipe",
                "source_overrides",
                "video_splits",
                "geometry",
            },
        )
        if type(cfg.source_recipe) is not str or not cfg.source_recipe.strip():
            raise ValueError("source_recipe must explicitly name a build configuration")
        if not isinstance(cfg.source_overrides, ListConfig):
            raise ValueError(
                "source_overrides must be a list of Hydra override strings"
            )
        overrides = tuple(cfg.source_overrides)
        if any(type(value) is not str for value in overrides):
            raise ValueError("source_overrides must contain Hydra override strings")
        if any(
            value.lstrip("+~").split("=", 1)[0].startswith("paths")
            for value in overrides
        ):
            raise ValueError("Declare shared path roots in paths, not source_overrides")
        if type(cfg.replay_count) is not int or cfg.replay_count < 0:
            raise ValueError("replay_count must be a nonnegative integer")
        splits = dict(cfg.video_splits)
        if not splits or set(splits.values()) != {"train", "val", "test"}:
            raise ValueError(
                "Explicit recording-disjoint train/val/test assignments required"
            )
        for video in splits:
            validate_id_component(video, field_name="video_id")
        exact_config_mapping(
            cfg.geometry,
            path="geometry",
            required_keys={
                "max_reprojection_px",
                "max_speed_mps",
                "xy_limit_m",
                "z_range_m",
            },
        )
        values = [
            cfg.geometry.max_reprojection_px,
            cfg.geometry.max_speed_mps,
            *cfg.geometry.xy_limit_m,
            *cfg.geometry.z_range_m,
        ]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("Geometry requires finite numeric limits")
        xy, z = tuple(cfg.geometry.xy_limit_m), tuple(cfg.geometry.z_range_m)
        if (
            len(xy) != 2
            or len(z) != 2
            or min(values[:2]) <= 0
            or min(xy) <= 0
            or z[0] >= z[1]
        ):
            raise ValueError("Invalid geometry limits")
        resolver = _resolver(cfg)
        return cls(
            resolver.resolve(PathRole.DATA, cfg.destination),
            resolver.resolve(PathRole.DATA, cfg.synthetic_source),
            cfg.replay_count,
            cfg.source_recipe,
            overrides,
            cfg.paths,
            splits,
            TriangulationSettings(float(values[0]), float(values[1]), xy, z),
        )


def validate_blcs_preparation_config(cfg: DictConfig) -> None:
    BLCSPreparationConfig.from_config(cfg)


def prepare_blcs(runtime: BLCSPreparationConfig) -> None:
    from src.tennis_scene.dataset_pipeline.blcs_training import export_blcs_training

    # Compose in the declared tennis-scene config namespace; never infer a recipe from media paths.
    if GlobalHydra.instance().is_initialized():
        source = compose(
            config_name=runtime.source_recipe, overrides=list(runtime.source_overrides)
        )
    else:
        with initialize_config_dir(
            config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"),
            version_base="1.3",
        ):
            source = compose(
                config_name=runtime.source_recipe,
                overrides=list(runtime.source_overrides),
            )
    source.paths = OmegaConf.create(OmegaConf.to_container(runtime.paths, resolve=True))
    export_blcs_training(
        source,
        runtime.destination,
        synthetic_source=runtime.synthetic_source,
        replay_count=runtime.replay_count,
        video_splits=runtime.video_splits,
        settings=runtime.geometry,
    )
