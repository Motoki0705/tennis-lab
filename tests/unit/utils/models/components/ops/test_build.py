"""Tests for root setup.py CUDA source preparation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.utils.models.components.ops import build as build_module

_LEGACY_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
_MODERN_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
_BUILD_MODULE_PATH = Path(build_module.__file__).resolve()
_PROJECT_ROOT = _BUILD_MODULE_PATH.parents[5]
_OPERATION_ENVIRONMENT_NAMES = (
    build_module.BUILD_CUDA_OPS,
    build_module.DINO_OPS_BUILD_CONFIG,
    "TENNIS_LAB_FORCE_MOE_REFERENCE",
    "TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE",
    "TENNIS_LAB_USE_TIME_LOCAL_CUDA",
)
_SPEC_LOAD_SCRIPT = """
import importlib.util
import json
import sys

build_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("tennis_lab_ops_build", build_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load ops build module from {build_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps([module.get_extensions(), module.get_cmdclass()], sort_keys=True))
"""


def _spec_loaded_build(
    tmp_path: Path,
    overrides: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in _OPERATION_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    environment.update(overrides)
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _SPEC_LOAD_SCRIPT,
            str(_BUILD_MODULE_PATH),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _build_mapping(project_root: Path) -> dict[str, object]:
    return {
        "paths": {
            "project_root": str(project_root),
            "data_root": "data",
            "checkpoint_root": "ckpt",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "third_party",
        },
        "source_role": "external_asset",
        "source": "DINO/ops/src",
        "destination_role": "cache",
        "destination": "dino_ops/src",
        "moe_bindings": "src/utils/models/components/ops/moe/csrc/moe.cpp",
        "moe_kernels": "src/utils/models/components/ops/moe/csrc/moe_cuda.cu",
        "time_local_bindings": (
            "src/utils/models/components/ops/time_local/csrc/time_local.cpp"
        ),
        "time_local_kernels": (
            "src/utils/models/components/ops/time_local/csrc/time_local_cuda.cu"
        ),
        "compressed_time_local_bindings": (
            "src/utils/models/components/ops/compressed_time_local/bindings.cpp"
        ),
        "compressed_time_local_kernels": (
            "src/utils/models/components/ops/compressed_time_local/kernels.cu"
        ),
    }


@pytest.mark.parametrize("raw", [None, "0"])
def test_setup_spec_load_is_standalone_when_build_is_disabled(
    tmp_path: Path,
    raw: str | None,
) -> None:
    overrides = {} if raw is None else {build_module.BUILD_CUDA_OPS: raw}

    completed = _spec_loaded_build(tmp_path, overrides)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [[], {}]


@pytest.mark.parametrize("raw", ["", "true", "yes", "2", " 0"])
def test_setup_spec_load_rejects_noncanonical_build_tokens(
    tmp_path: Path,
    raw: str,
) -> None:
    completed = _spec_loaded_build(
        tmp_path,
        {build_module.BUILD_CUDA_OPS: raw},
    )

    assert completed.returncode != 0
    assert "must be exactly '0' or '1'" in completed.stderr


@pytest.mark.parametrize("raw_json", [None, ""])
def test_enabled_setup_spec_load_requires_nonempty_build_json(
    tmp_path: Path,
    raw_json: str | None,
) -> None:
    overrides = {build_module.BUILD_CUDA_OPS: "1"}
    if raw_json is not None:
        overrides[build_module.DINO_OPS_BUILD_CONFIG] = raw_json

    completed = _spec_loaded_build(tmp_path, overrides)

    assert completed.returncode != 0
    assert build_module.DINO_OPS_BUILD_CONFIG in completed.stderr


@pytest.mark.parametrize(
    ("raw_json", "message"),
    [
        ("{", "must contain valid JSON"),
        ("[]", "must decode to a JSON object"),
        ("{}", ".paths must be a JSON object"),
    ],
)
def test_enabled_setup_spec_load_rejects_invalid_bootstrap_json(
    tmp_path: Path,
    raw_json: str,
    message: str,
) -> None:
    completed = _spec_loaded_build(
        tmp_path,
        {
            build_module.BUILD_CUDA_OPS: "1",
            build_module.DINO_OPS_BUILD_CONFIG: raw_json,
        },
    )

    assert completed.returncode != 0
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("project_root", "message"),
    [
        (".", "must be absolute"),
        ("{unresolved}", "must already be resolved"),
        ("{missing}", "Canonical operation contract was not found"),
    ],
)
def test_enabled_setup_spec_load_rejects_invalid_bootstrap_root(
    tmp_path: Path,
    project_root: str,
    message: str,
) -> None:
    rendered_root = project_root.format(
        unresolved=tmp_path / "child" / "..",
        missing=tmp_path / "missing-project",
    )
    raw_json = json.dumps({"paths": {"project_root": rendered_root}})

    completed = _spec_loaded_build(
        tmp_path,
        {
            build_module.BUILD_CUDA_OPS: "1",
            build_module.DINO_OPS_BUILD_CONFIG: raw_json,
        },
    )

    assert completed.returncode != 0
    assert message in completed.stderr


def test_enabled_setup_spec_load_delegates_canonical_validation(
    tmp_path: Path,
) -> None:
    mapping = _build_mapping(_PROJECT_ROOT)
    mapping["source_role"] = "project"

    completed = _spec_loaded_build(
        tmp_path,
        {
            build_module.BUILD_CUDA_OPS: "1",
            build_module.DINO_OPS_BUILD_CONFIG: json.dumps(mapping),
        },
    )

    assert completed.returncode != 0
    assert "dino_ops_build.source_role" in completed.stderr


def test_enabled_setup_spec_load_delegates_environment_conflicts(
    tmp_path: Path,
) -> None:
    completed = _spec_loaded_build(
        tmp_path,
        {
            build_module.BUILD_CUDA_OPS: "1",
            build_module.DINO_OPS_BUILD_CONFIG: json.dumps(
                _build_mapping(_PROJECT_ROOT)
            ),
            "TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE": "1",
            "TENNIS_LAB_USE_TIME_LOCAL_CUDA": "1",
        },
    )

    assert completed.returncode != 0
    assert "conflicts" in completed.stderr


def test_operation_loader_rejects_preloaded_module_from_another_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_root = tmp_path / "configured-project"
    unexpected_module = ModuleType("src.utils.configuration.operations")
    unexpected_module.__file__ = str(tmp_path / "other-project/operations.py")
    monkeypatch.setitem(
        sys.modules,
        "src.utils.configuration.operations",
        unexpected_module,
    )

    with pytest.raises(RuntimeError, match="unexpected module path"):
        build_module._load_operations(expected_root)


def test_prepare_dino_ops_sources_patches_copy_only(tmp_path: Path) -> None:
    source = tmp_path / "third_party_src"
    cuda_source = source / "cuda/ms_deform_attn_cuda.cu"
    cuda_source.parent.mkdir(parents=True)
    original = f"{_LEGACY_DISPATCH}\n{_LEGACY_DISPATCH}\n"
    cuda_source.write_text(original)
    destination = tmp_path / "build_src"

    result = build_module._prepare_dino_ops_sources(source, destination)

    assert result == destination
    assert cuda_source.read_text() == original
    generated = (destination / "cuda/ms_deform_attn_cuda.cu").read_text()
    assert _LEGACY_DISPATCH not in generated
    assert generated.count(_MODERN_DISPATCH) == 2


def test_prepare_dino_ops_sources_requires_initialized_submodule(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="configured DINO external asset"):
        build_module._prepare_dino_ops_sources(
            tmp_path / "missing",
            tmp_path / "build",
        )
