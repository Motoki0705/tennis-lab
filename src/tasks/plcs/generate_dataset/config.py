"""Strict PLCS dataset-generation configuration and role-based path resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

from omegaconf import DictConfig, OmegaConf

import src.tasks.plcs.configuration_contracts as configuration_contracts
from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.plcs.court_keypoint_contract import PLCSCourtKeypointRuntimeConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.configuration.paths import PathRole
from src.utils.device import DeviceSelectionError, resolve_device
from src.utils.hydra import register_boundary_validator


def _reject_unknown(
    mapping: object, allowed: set[str], *, path: str
) -> Mapping[str, object]:
    resolved: Mapping[str, object] = as_config_mapping(mapping, path=path)
    unknown = sorted(set(resolved) - allowed)
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): {', '.join(f'{path}.{key}' for key in unknown)}."
        )
    return resolved


@dataclass(frozen=True, slots=True)
class PLCSExternalAssets:
    """Resolved external assets required by generation workers."""

    smplh_model_path: Path


@dataclass(frozen=True, slots=True)
class PLCSGenerationConfig:
    """Validated generation boundary with a fully resolved worker config."""

    config: DictConfig
    court_keypoint_contract: CourtKeypointContract
    output_dir: Path
    device: str
    seed: int
    num_workers: int
    num_scenes: int
    generation_mode: str
    external_assets: PLCSExternalAssets
    train_ratio: float
    val_ratio: float
    test_ratio: float

    OUTPUT_ROLE: ClassVar[PathRole] = PathRole.DATA

    @classmethod
    def from_config(cls, value: object) -> PLCSGenerationConfig:
        if not isinstance(value, DictConfig):
            raise ConfigurationTypeError(
                "PLCS generation boundary requires DictConfig."
            )
        components = configuration_contracts.PLCSGenerationComponents.from_config(
            value
        )
        path_config = components.paths
        root = as_config_mapping(
            OmegaConf.to_container(value, resolve=True), path="configuration"
        )
        root = _reject_unknown(
            root,
            {
                "court_keypoints",
                "generation",
                "paths",
                "external_assets",
                "simulation",
                "camera",
                "motion_sources",
                "run",
            },
            path="configuration",
        )
        run = _reject_unknown(
            require_config_mapping(root, "run", path="configuration"),
            {
                "output_dir",
                "seed",
                "device",
                "num_workers",
                "train_ratio",
                "val_ratio",
                "test_ratio",
            },
            path="run",
        )
        mode = components.mode
        output_relative = cast(
            "str", require_config_value(run, "output_dir", str, path="run")
        )
        output_dir = path_config.resolver.resolve(PathRole.DATA, output_relative)
        requested_device = cast(
            "str", require_config_value(run, "device", str, path="run")
        )
        try:
            device = str(resolve_device(requested_device))
        except DeviceSelectionError as error:
            raise SemanticConfigurationError(
                f"run.device is not an available device: {requested_device!r}."
            ) from error
        if device != "cpu":
            raise SemanticConfigurationError(
                "run.device must resolve to 'cpu' for parallel PLCS generation."
            )
        workers = cast("int", require_config_value(run, "num_workers", int, path="run"))
        simulation = _reject_unknown(
            require_config_mapping(root, "simulation", path="configuration"),
            {"num_scenes"},
            path="simulation",
        )
        scenes = cast(
            "int",
            require_config_value(simulation, "num_scenes", int, path="simulation"),
        )
        if workers < 1 or scenes < 1:
            raise SemanticConfigurationError(
                "run.num_workers and simulation.num_scenes must be positive."
            )
        ratios = tuple(
            float(
                cast(
                    "float | int",
                    require_config_value(run, key, (float, int), path="run"),
                )
            )
            for key in ("train_ratio", "val_ratio", "test_ratio")
        )
        if (
            any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios)
            or abs(sum(ratios) - 1.0) > 1e-8
        ):
            raise SemanticConfigurationError(
                "Generation split ratios must be non-negative and sum to 1."
            )

        resolved = _resolve_generation_paths(value, components=components)
        resolved.run.output_dir = str(output_dir)
        resolved.run.device = device
        return cls(
            config=resolved,
            court_keypoint_contract=(
                PLCSCourtKeypointRuntimeConfig.from_config(value).contract
            ),
            output_dir=output_dir,
            device=device,
            seed=cast("int", require_config_value(run, "seed", int, path="run")),
            num_workers=workers,
            num_scenes=scenes,
            generation_mode=mode,
            external_assets=PLCSExternalAssets(
                smplh_model_path=Path(str(resolved.external_assets.smplh_model_path))
            ),
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
        )


def _validate_boundary(config: DictConfig) -> None:
    PLCSGenerationConfig.from_config(config)


register_boundary_validator("plcs.generate_dataset", _validate_boundary)

__all__ = [
    "PLCSGenerationConfig",
    "resolve_generation_paths",
]


def resolve_generation_paths(value: DictConfig) -> DictConfig:
    """Return a resolved copy for generation consumers, including chunk workers."""
    components = configuration_contracts.PLCSGenerationComponents.from_config(value)
    return _resolve_generation_paths(value, components=components)


def _resolve_generation_paths(
    value: DictConfig,
    *,
    components: configuration_contracts.PLCSGenerationComponents,
) -> DictConfig:
    """Resolve a generation config after its shared contract is validated."""
    path_config = components.paths
    root = as_config_mapping(
        OmegaConf.to_container(value, resolve=True), path="configuration"
    )
    external_assets = require_config_mapping(
        root, "external_assets", path="configuration"
    )
    _reject_unknown(external_assets, {"smplh_model_path"}, path="external_assets")
    smplh_relative = cast(
        "str",
        require_config_value(
            external_assets, "smplh_model_path", str, path="external_assets"
        ),
    )
    container = OmegaConf.to_container(value, resolve=True)
    resolved = OmegaConf.create(container)
    if not isinstance(resolved, DictConfig):
        raise ConfigurationTypeError(
            "PLCS generation config must resolve to DictConfig."
        )
    resolved.external_assets.smplh_model_path = str(
        path_config.resolver.resolve(PathRole.EXTERNAL_ASSET, smplh_relative)
    )
    for category, source in resolved.motion_sources.items():
        source_mapping = as_config_mapping(source, path=f"motion_sources.{category}")
        paths = cast(
            "Sequence[str]",
            source_mapping["paths"],
        )
        source.paths = [
            str(path_config.resolver.resolve(PathRole.EXTERNAL_ASSET, path))
            for path in paths
        ]
    return resolved
