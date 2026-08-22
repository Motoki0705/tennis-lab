"""Strict execution-input contracts for optional model operations and builds."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from src.utils.configuration.errors import (
    ConfigurationTypeError,
    SemanticConfigurationError,
)
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots
from src.utils.configuration.schema import ConfigField, StrictConfigSchema

BUILD_CUDA_OPS = "TENNIS_LAB_BUILD_CUDA_OPS"
CUDA_OPS_BUILD_TARGET = "TENNIS_LAB_CUDA_OPS_BUILD_TARGET"
FORCE_MOE_REFERENCE = "TENNIS_LAB_FORCE_MOE_REFERENCE"
FORCE_TIME_LOCAL_REFERENCE = "TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE"
USE_TIME_LOCAL_CUDA = "TENNIS_LAB_USE_TIME_LOCAL_CUDA"
DINO_OPS_BUILD_CONFIG = "TENNIS_LAB_DINO_OPS_BUILD_CONFIG"

_BOOLEAN_NAMES = (
    BUILD_CUDA_OPS,
    FORCE_MOE_REFERENCE,
    FORCE_TIME_LOCAL_REFERENCE,
    USE_TIME_LOCAL_CUDA,
)
_RUNTIME_BOOLEAN_NAMES = (
    FORCE_MOE_REFERENCE,
    FORCE_TIME_LOCAL_REFERENCE,
    USE_TIME_LOCAL_CUDA,
)
_OPERATION_ENVIRONMENT_DEFAULTS = {name: "0" for name in _RUNTIME_BOOLEAN_NAMES}
_OPERATION_ENVIRONMENT_SCHEMA = StrictConfigSchema(
    name="operation_environment",
    fields={
        BUILD_CUDA_OPS: ConfigField.of(str, required=False),
        CUDA_OPS_BUILD_TARGET: ConfigField.of(str, required=False),
        **{name: ConfigField.of(str) for name in _RUNTIME_BOOLEAN_NAMES},
        DINO_OPS_BUILD_CONFIG: ConfigField.of(str, required=False),
    },
)


def _parse_build_cuda_ops(raw: str | None) -> bool:
    """Delegate the optional build switch to the standalone build authority."""
    build_module = importlib.import_module("src.utils.models.components.ops.build")
    parse_build_cuda_ops = cast(
        Callable[[str | None], bool],
        build_module.parse_build_cuda_ops,
    )

    try:
        return parse_build_cuda_ops(raw)
    except ValueError as error:
        raise SemanticConfigurationError(str(error)) from error


def _parse_cuda_ops_build_target(raw: str | None) -> str:
    """Delegate the extension selection to the standalone build authority."""
    build_module = importlib.import_module("src.utils.models.components.ops.build")
    parse_target = cast(
        Callable[[str | None], str],
        build_module.parse_cuda_ops_build_target,
    )
    try:
        return parse_target(raw)
    except ValueError as error:
        raise SemanticConfigurationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class OperationEnvironmentConfig:
    """Validated process inputs controlling optional CUDA operation routes."""

    build_cuda_ops: bool
    cuda_ops_build_target: str
    force_moe_reference: bool
    force_time_local_reference: bool
    use_time_local_cuda: bool
    dino_ops_build_config: str | None = field(
        metadata={
            "configuration_required": False,
            "configuration_absence_policy": "optional-input-omitted-as-none",
        }
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> OperationEnvironmentConfig:
        """Reject unknown names and accept only exact ``"0"``/``"1"`` tokens."""
        validated = _OPERATION_ENVIRONMENT_SCHEMA.validate(values)
        parsed = {
            BUILD_CUDA_OPS: _parse_build_cuda_ops(
                cast(str, validated[BUILD_CUDA_OPS])
                if BUILD_CUDA_OPS in validated
                else None
            )
        }
        build_target = _parse_cuda_ops_build_target(
            cast(str, validated[CUDA_OPS_BUILD_TARGET])
            if CUDA_OPS_BUILD_TARGET in validated
            else None
        )
        for name in _RUNTIME_BOOLEAN_NAMES:
            raw = cast(str, validated[name])
            if raw not in {"0", "1"}:
                raise SemanticConfigurationError(
                    f"{name} must be exactly '0' or '1'; got {raw!r}."
                )
            parsed[name] = raw == "1"
        if parsed[FORCE_TIME_LOCAL_REFERENCE] and parsed[USE_TIME_LOCAL_CUDA]:
            raise SemanticConfigurationError(
                f"{FORCE_TIME_LOCAL_REFERENCE}=1 conflicts with "
                f"{USE_TIME_LOCAL_CUDA}=1."
            )
        build_json = (
            cast(str, validated[DINO_OPS_BUILD_CONFIG])
            if DINO_OPS_BUILD_CONFIG in validated
            else None
        )
        if build_json == "":
            raise SemanticConfigurationError(
                f"{DINO_OPS_BUILD_CONFIG} must be a non-empty JSON string."
            )
        return cls(
            build_cuda_ops=parsed[BUILD_CUDA_OPS],
            cuda_ops_build_target=build_target,
            force_moe_reference=parsed[FORCE_MOE_REFERENCE],
            force_time_local_reference=parsed[FORCE_TIME_LOCAL_REFERENCE],
            use_time_local_cuda=parsed[USE_TIME_LOCAL_CUDA],
            dino_ops_build_config=build_json,
        )

    @classmethod
    def from_process_environment(cls) -> OperationEnvironmentConfig:
        """Read the one documented allow-list from the process environment."""
        selected = dict(_OPERATION_ENVIRONMENT_DEFAULTS)
        selected.update(
            {name: os.environ[name] for name in _BOOLEAN_NAMES if name in os.environ}
        )
        if DINO_OPS_BUILD_CONFIG in os.environ:
            selected[DINO_OPS_BUILD_CONFIG] = os.environ[DINO_OPS_BUILD_CONFIG]
        if CUDA_OPS_BUILD_TARGET in os.environ:
            selected[CUDA_OPS_BUILD_TARGET] = os.environ[CUDA_OPS_BUILD_TARGET]
        return cls.from_mapping(selected)

    def require_dino_build_config(self, *, repository_root: Path) -> DinoOpsBuildConfig:
        """Parse the required build JSON when CUDA extension building is enabled."""
        if self.dino_ops_build_config is None:
            raise SemanticConfigurationError(
                f"{DINO_OPS_BUILD_CONFIG} is required when {BUILD_CUDA_OPS}=1."
            )
        return DinoOpsBuildConfig.from_json(
            self.dino_ops_build_config,
            repository_root=repository_root,
        )


_ROOTS_SCHEMA = StrictConfigSchema(
    name="dino_ops_build.paths",
    fields={f"{role.value}_root": ConfigField.of(str) for role in PathRole},
)
_DINO_BUILD_SCHEMA = StrictConfigSchema(
    name="dino_ops_build",
    fields={
        "paths": ConfigField.mapping(_ROOTS_SCHEMA),
        "source_role": ConfigField.of(str),
        "source": ConfigField.of(str),
        "destination_role": ConfigField.of(str),
        "destination": ConfigField.of(str),
        "moe_bindings": ConfigField.of(str),
        "moe_kernels": ConfigField.of(str),
        "time_local_bindings": ConfigField.of(str),
        "time_local_kernels": ConfigField.of(str),
        "compressed_time_local_bindings": ConfigField.of(str),
        "compressed_time_local_kernels": ConfigField.of(str),
    },
)


@dataclass(frozen=True, slots=True)
class DinoOpsBuildConfig:
    """All role-resolved source paths consumed by the CUDA build boundary."""

    resolver: PathResolver
    source: Path
    destination: Path
    destination_role: PathRole
    moe_bindings: Path
    moe_kernels: Path
    time_local_bindings: Path
    time_local_kernels: Path
    compressed_time_local_bindings: Path
    compressed_time_local_kernels: Path

    def require_inputs(self) -> None:
        """Fail before importing the build toolchain when any source is absent."""
        if not self.source.is_dir():
            raise FileNotFoundError(
                f"DINO operation source directory is missing: {self.source}"
            )
        for path in (
            self.moe_bindings,
            self.moe_kernels,
            self.time_local_bindings,
            self.time_local_kernels,
            self.compressed_time_local_bindings,
            self.compressed_time_local_kernels,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"CUDA operation source file is missing: {path}"
                )

    @classmethod
    def from_json(cls, raw: str, *, repository_root: Path) -> DinoOpsBuildConfig:
        """Decode a strict JSON object without reconstructing any root or path."""
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SemanticConfigurationError(
                f"{DINO_OPS_BUILD_CONFIG} must contain valid JSON."
            ) from error
        if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
            raise ConfigurationTypeError(
                f"{DINO_OPS_BUILD_CONFIG} must decode to an object with string keys."
            )
        return cls.from_mapping(
            cast(Mapping[str, object], value), repository_root=repository_root
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        repository_root: Path,
    ) -> DinoOpsBuildConfig:
        """Validate exact keys, canonical roles, and root-contained fragments."""
        validated = _DINO_BUILD_SCHEMA.validate(value)
        raw_paths = validated["paths"]
        if not isinstance(raw_paths, Mapping):
            raise AssertionError("DINO build paths were not validated as a mapping.")
        raw_project_root = raw_paths["project_root"]
        if type(raw_project_root) is not str:
            raise AssertionError("DINO build project_root was not validated as str.")
        serialized_project_root = Path(raw_project_root)
        if not serialized_project_root.is_absolute():
            raise SemanticConfigurationError(
                "dino_ops_build.paths.project_root must be an explicit absolute path."
            )
        if serialized_project_root.resolve(strict=False) != serialized_project_root:
            raise SemanticConfigurationError(
                "dino_ops_build.paths.project_root must be a resolved absolute path."
            )
        if not repository_root.is_absolute():
            raise SemanticConfigurationError(
                "repository_root must be an explicit absolute path."
            )
        if repository_root.resolve(strict=False) != repository_root:
            raise SemanticConfigurationError(
                "repository_root must be a resolved absolute path."
            )
        if serialized_project_root != repository_root:
            raise SemanticConfigurationError(
                "dino_ops_build.paths.project_root must exactly match repository_root."
            )
        roots = RuntimePathRoots.from_mapping(
            raw_paths, repository_root=repository_root
        )
        resolver = PathResolver(roots)
        source_role = cast(str, validated["source_role"])
        if source_role != PathRole.EXTERNAL_ASSET.value:
            raise SemanticConfigurationError(
                "dino_ops_build.source_role must be 'external_asset'."
            )
        raw_destination_role = cast(str, validated["destination_role"])
        try:
            destination_role = PathRole(raw_destination_role)
        except ValueError as error:
            raise SemanticConfigurationError(
                f"dino_ops_build.destination_role is unsupported: {raw_destination_role!r}."
            ) from error
        if destination_role not in {PathRole.CACHE, PathRole.ARTIFACT}:
            raise SemanticConfigurationError(
                "dino_ops_build.destination_role must be 'cache' or 'artifact'."
            )

        def resolve(role: PathRole, key: str) -> Path:
            fragment = cast(str, validated[key])
            if not fragment or Path(fragment) in {Path("."), Path("..")}:
                raise SemanticConfigurationError(
                    f"dino_ops_build.{key} must be a non-empty child path."
                )
            resolved: Path = resolver.resolve(role, fragment)
            return resolved

        source = resolve(PathRole.EXTERNAL_ASSET, "source")
        destination = resolve(destination_role, "destination")
        if source == destination:
            raise SemanticConfigurationError(
                "DINO operation source and build destination must differ."
            )
        return cls(
            resolver=resolver,
            source=source,
            destination=destination,
            destination_role=destination_role,
            moe_bindings=resolve(PathRole.PROJECT, "moe_bindings"),
            moe_kernels=resolve(PathRole.PROJECT, "moe_kernels"),
            time_local_bindings=resolve(PathRole.PROJECT, "time_local_bindings"),
            time_local_kernels=resolve(PathRole.PROJECT, "time_local_kernels"),
            compressed_time_local_bindings=resolve(
                PathRole.PROJECT, "compressed_time_local_bindings"
            ),
            compressed_time_local_kernels=resolve(
                PathRole.PROJECT, "compressed_time_local_kernels"
            ),
        )


def operation_environment() -> OperationEnvironmentConfig:
    """Return the sole validated process-environment contract for model ops."""
    return OperationEnvironmentConfig.from_process_environment()


__all__ = [
    "BUILD_CUDA_OPS",
    "CUDA_OPS_BUILD_TARGET",
    "DINO_OPS_BUILD_CONFIG",
    "DinoOpsBuildConfig",
    "FORCE_MOE_REFERENCE",
    "FORCE_TIME_LOCAL_REFERENCE",
    "OperationEnvironmentConfig",
    "USE_TIME_LOCAL_CUDA",
    "operation_environment",
]
