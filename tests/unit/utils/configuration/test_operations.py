"""Strict operation-environment and CUDA-build input contracts."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from src.utils.configuration import (
    BUILD_CUDA_OPS,
    CUDA_OPS_BUILD_TARGET,
    DINO_OPS_BUILD_CONFIG,
    ConfigurationError,
    DinoOpsBuildConfig,
    OperationEnvironmentConfig,
)

parse_build_cuda_ops = cast(
    Callable[[str | None], bool],
    importlib.import_module(
        "src.utils.models.components.ops.build"
    ).parse_build_cuda_ops,
)
parse_cuda_ops_build_target = cast(
    Callable[[str | None], str],
    importlib.import_module(
        "src.utils.models.components.ops.build"
    ).parse_cuda_ops_build_target,
)


def _build_mapping(root: Path) -> dict[str, object]:
    return {
        "paths": {
            "project_root": str(root.resolve()),
            "data_root": "data",
            "checkpoint_root": "ckpt",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "third_party",
        },
        "source_role": "external_asset",
        "source": "DINO/models/dino/ops/src",
        "destination_role": "cache",
        "destination": "dino_ops/src",
        "compressed_time_local_bindings": (
            "src/utils/models/components/ops/compressed_time_local/bindings.cpp"
        ),
        "compressed_time_local_kernels": (
            "src/utils/models/components/ops/compressed_time_local/kernels.cu"
        ),
    }


def _operation_environment(**overrides: str) -> dict[str, str]:
    values = {BUILD_CUDA_OPS: "0"}
    values.update(overrides)
    return values


@pytest.mark.parametrize("raw", ["", "true", "yes", "2", " 0"])
def test_boolean_environment_tokens_are_exact(raw: str) -> None:
    with pytest.raises(ConfigurationError, match="exactly '0' or '1'"):
        OperationEnvironmentConfig.from_mapping(
            _operation_environment(**{BUILD_CUDA_OPS: raw})
        )


@pytest.mark.parametrize(
    "name",
    [
        "TENNIS_LAB_TYPO",
        "TENNIS_LAB_FORCE_MOE_REFERENCE",
        "TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE",
        "TENNIS_LAB_USE_TIME_LOCAL_CUDA",
    ],
)
def test_operation_environment_rejects_unknown_inputs(name: str) -> None:
    with pytest.raises(ConfigurationError, match="Unknown configuration"):
        OperationEnvironmentConfig.from_mapping({name: "1"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, False), ("0", False), ("1", True)],
)
def test_runtime_environment_reuses_standalone_build_switch_authority(
    raw: str | None,
    expected: bool,
) -> None:
    values = _operation_environment()
    if raw is None:
        del values[BUILD_CUDA_OPS]
    else:
        values[BUILD_CUDA_OPS] = raw

    assert parse_build_cuda_ops(raw) is expected
    assert OperationEnvironmentConfig.from_mapping(values).build_cuda_ops is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "all"),
        ("all", "all"),
        ("compressed_time_local", "compressed_time_local"),
    ],
)
def test_runtime_environment_reuses_standalone_build_target_authority(
    raw: str | None,
    expected: str,
) -> None:
    values = _operation_environment()
    if raw is not None:
        values[CUDA_OPS_BUILD_TARGET] = raw

    environment = OperationEnvironmentConfig.from_mapping(values)

    assert parse_cuda_ops_build_target(raw) == expected
    assert environment.cuda_ops_build_target == expected


def test_runtime_environment_rejects_unknown_build_target() -> None:
    with pytest.raises(ConfigurationError, match=CUDA_OPS_BUILD_TARGET):
        OperationEnvironmentConfig.from_mapping(
            _operation_environment(**{CUDA_OPS_BUILD_TARGET: "compressed"})
        )


def test_enabled_build_requires_strict_json_contract(tmp_path: Path) -> None:
    environment = OperationEnvironmentConfig.from_mapping(
        _operation_environment(**{BUILD_CUDA_OPS: "1"})
    )
    with pytest.raises(ConfigurationError, match=DINO_OPS_BUILD_CONFIG):
        environment.require_dino_build_config(repository_root=tmp_path.resolve())


def test_enabled_build_uses_serialized_absolute_project_root(tmp_path: Path) -> None:
    environment = OperationEnvironmentConfig.from_mapping(
        _operation_environment(
            **{
                BUILD_CUDA_OPS: "1",
                DINO_OPS_BUILD_CONFIG: json.dumps(_build_mapping(tmp_path)),
            }
        )
    )

    config = environment.require_dino_build_config(repository_root=tmp_path.resolve())

    assert config.resolver.roots.project_root == tmp_path.resolve()


def test_dino_build_contract_resolves_all_roles(tmp_path: Path) -> None:
    config = DinoOpsBuildConfig.from_mapping(
        _build_mapping(tmp_path), repository_root=tmp_path.resolve()
    )

    assert config.source == tmp_path / "third_party/DINO/models/dino/ops/src"
    assert config.destination == tmp_path / ".cache/dino_ops/src"
    assert config.compressed_time_local_kernels == (
        tmp_path / "src/utils/models/components/ops/compressed_time_local/kernels.cu"
    )


@pytest.mark.parametrize(
    ("serialized_root", "message"),
    [
        (".", "explicit absolute"),
        ("{other}", "exactly match"),
        ("{unresolved}", "resolved absolute"),
    ],
)
def test_dino_build_contract_rejects_noncanonical_serialized_project_root(
    tmp_path: Path,
    serialized_root: str,
    message: str,
) -> None:
    mapping = _build_mapping(tmp_path)
    paths = mapping["paths"]
    assert isinstance(paths, dict)
    paths["project_root"] = serialized_root.format(
        other=tmp_path.parent / "other-project",
        unresolved=tmp_path / "child" / "..",
    )

    with pytest.raises(ConfigurationError, match=message):
        DinoOpsBuildConfig.from_mapping(
            mapping,
            repository_root=tmp_path.resolve(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"typo": "x"}, "Unknown"),
        ({"source_role": "project"}, "source_role"),
        ({"destination_role": "project"}, "destination_role"),
        ({"source": "/tmp/outside"}, "relative"),
        ({"destination": "../outside"}, "escapes"),
        ({"compressed_time_local_bindings": 3}, "expected str"),
        ({"moe_bindings": "legacy"}, "Unknown"),
        ({"time_local_bindings": "legacy"}, "Unknown"),
    ],
)
def test_dino_build_contract_rejects_invalid_inputs(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    mapping = _build_mapping(tmp_path)
    mapping.update(mutation)
    with pytest.raises(ConfigurationError, match=message):
        DinoOpsBuildConfig.from_mapping(mapping, repository_root=tmp_path.resolve())
