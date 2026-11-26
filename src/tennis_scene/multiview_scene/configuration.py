"""Parameters borrowed from a saved standard batch for the multiview body recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.submodules.configuration import BodyRuntimeConfig, SmplAssetPaths
from src.tasks.person_tracking.features import FeatureConfig
from src.tennis_scene.configuration import _PLACEMENT_SCHEMA
from src.tennis_scene.motion_alignment.temporal import TemporalPlacementConfig
from src.tennis_scene.pipeline.model_assets import BodyModelConfig
from src.tennis_scene.pipeline.recipes import STANDARD_RECIPE
from src.utils.configuration import (
    ConfigField,
    MissingConfigurationKeyError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT

_SETTINGS = StrictConfigSchema(
    name="multiview_scene.saved_batch",
    fields={
        "device": ConfigField.of(str),
        "people_models.hmr2_checkpoint": ConfigField.of(str),
        "people_models.gvhmr_checkpoint": ConfigField.of(str),
        "people_models.body_models_dir": ConfigField.of(str),
        "people_models.runtime.tracking.bbox_enlarge": ConfigField.of(float, int),
        "frame_sampling.max_frames": ConfigField.of(int),
        "person_observations.visibility_threshold": ConfigField.of(float, int),
        "player_reconstruction.reprojection_px": ConfigField.of(float, int),
    },
)


@dataclass(frozen=True)
class MultiviewSceneConfig:
    people: BodyModelConfig
    crops: FeatureConfig
    sampling_max_frames: int
    human_vis_threshold: float
    player_placement: TemporalPlacementConfig
    player_reprojection_px: float

    @classmethod
    def from_batch_yaml(cls, text: str) -> MultiviewSceneConfig:
        stored = OmegaConf.create(text)
        if not isinstance(stored, DictConfig):
            raise ValueError("Expected a saved pipeline configuration mapping")
        if (
            OmegaConf.select(stored, "recipe", default=STANDARD_RECIPE)
            != STANDARD_RECIPE
        ):
            raise ValueError("Multiview GVHMR requires progress from recipe=standard")

        def required(key: str) -> object:
            value: object = OmegaConf.select(stored, key, throw_on_missing=True)
            if value is None:
                raise MissingConfigurationKeyError(f"Saved batch requires {key}")
            return value

        def section(key: str) -> Mapping[str, object]:
            value = required(key)
            if not isinstance(value, DictConfig):
                raise ValueError(f"Saved batch {key} must be a mapping")
            return cast(
                Mapping[str, object], OmegaConf.to_container(value, resolve=True)
            )

        settings = _SETTINGS.validate({key: required(key) for key in _SETTINGS.fields})
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(
                section("paths"), repository_root=PROJECT_ROOT
            )
        )
        runtime_prefix = "people_models.runtime"
        body_runtime = BodyRuntimeConfig.from_mapping(
            {
                "device": required(f"{runtime_prefix}.device"),
                "static_cam": required(f"{runtime_prefix}.static_cam"),
                "hmr2": section(f"{runtime_prefix}.hmr2"),
                "gvhmr": section(f"{runtime_prefix}.gvhmr"),
            },
            resolver=resolver,
        )
        if body_runtime.device != settings["device"]:
            raise ValueError("Body model device must equal the saved pipeline device")
        body = BodyModelConfig(
            hmr2_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT,
                cast(str, settings["people_models.hmr2_checkpoint"]),
            ),
            gvhmr_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT,
                cast(str, settings["people_models.gvhmr_checkpoint"]),
            ),
            body_models_dir=resolver.resolve(
                PathRole.CHECKPOINT,
                cast(str, settings["people_models.body_models_dir"]),
            ),
            smpl_assets=SmplAssetPaths.from_mapping(
                section("people_models.smpl_assets"), resolver=resolver
            ),
            runtime=body_runtime,
        )
        max_frames = cast(int, settings["frame_sampling.max_frames"])
        visibility = float(
            cast(float, settings["person_observations.visibility_threshold"])
        )
        reprojection = float(
            cast(float, settings["player_reconstruction.reprojection_px"])
        )
        if (
            max_frames <= 0
            or not 0 <= visibility <= 1
            or not isfinite(reprojection)
            or reprojection <= 0
        ):
            raise ValueError(
                "Invalid saved frame limit, visibility threshold or reprojection threshold"
            )
        placement = _PLACEMENT_SCHEMA.validate(
            section("player_reconstruction.placement")
        )
        return cls(
            people=body,
            crops=FeatureConfig(
                bbox_enlarge=float(
                    cast(float, settings[f"{runtime_prefix}.tracking.bbox_enlarge"])
                )
            ),
            sampling_max_frames=max_frames,
            human_vis_threshold=visibility,
            player_placement=TemporalPlacementConfig(**cast(dict[str, Any], placement)),
            player_reprojection_px=reprojection,
        )
