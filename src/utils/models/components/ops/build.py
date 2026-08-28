from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

BUILD_CUDA_OPS = "TENNIS_LAB_BUILD_CUDA_OPS"
CUDA_OPS_BUILD_TARGET = "TENNIS_LAB_CUDA_OPS_BUILD_TARGET"
DINO_OPS_BUILD_CONFIG = "TENNIS_LAB_DINO_OPS_BUILD_CONFIG"

ALL_CUDA_OPS = "all"
COMPRESSED_TIME_LOCAL_CUDA_OP = "compressed_time_local"
_CUDA_OPS_BUILD_TARGETS = {
    ALL_CUDA_OPS,
    COMPRESSED_TIME_LOCAL_CUDA_OP,
}

_OPERATIONS_MODULE = "src.utils.configuration.operations"
_OPERATIONS_RELATIVE_PATH = Path("src/utils/configuration/operations.py")
_BOOTSTRAP_NAMESPACE_PATHS = (
    ("src", Path("src")),
    ("src.utils", Path("src/utils")),
    ("src.utils.configuration", Path("src/utils/configuration")),
    ("src.utils.models", Path("src/utils/models")),
    ("src.utils.models.components", Path("src/utils/models/components")),
    ("src.utils.models.components.ops", Path("src/utils/models/components/ops")),
)

_OLD_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
_NEW_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"


def parse_build_cuda_ops(raw: str | None, /) -> bool:
    """Parse the one packaging-safe CUDA build switch without importing ``src``."""
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{BUILD_CUDA_OPS} must be exactly '0' or '1'; got {raw!r}.")


def should_build_cuda_ops() -> bool:
    """Return the strict build switch without importing the source package."""
    try:
        raw = os.environ[BUILD_CUDA_OPS]
    except KeyError:
        raw = None
    return parse_build_cuda_ops(raw)


def parse_cuda_ops_build_target(raw: str | None, /) -> str:
    """Parse the exact extension selection used by the packaging boundary."""
    if raw is None:
        return ALL_CUDA_OPS
    if raw in _CUDA_OPS_BUILD_TARGETS:
        return raw
    choices = ", ".join(sorted(_CUDA_OPS_BUILD_TARGETS))
    raise ValueError(f"{CUDA_OPS_BUILD_TARGET} must be one of: {choices}; got {raw!r}.")


def selected_cuda_ops_build_target() -> str:
    """Return the validated extension selection without importing ``src``."""
    return parse_cuda_ops_build_target(os.environ.get(CUDA_OPS_BUILD_TARGET))


def _required_build_json() -> str:
    if DINO_OPS_BUILD_CONFIG not in os.environ:
        raise ValueError(
            f"{DINO_OPS_BUILD_CONFIG} is required when {BUILD_CUDA_OPS}=1."
        )
    raw = os.environ[DINO_OPS_BUILD_CONFIG]
    if not raw:
        raise ValueError(f"{DINO_OPS_BUILD_CONFIG} must be a non-empty JSON string.")
    return raw


def _bootstrap_project_root(raw: str) -> Path:
    """Read only the absolute project root needed to import the full contract."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{DINO_OPS_BUILD_CONFIG} must contain valid JSON.") from error
    if type(value) is not dict:
        raise TypeError(f"{DINO_OPS_BUILD_CONFIG} must decode to a JSON object.")
    if "paths" not in value or type(value["paths"]) is not dict:
        raise TypeError(
            f"{DINO_OPS_BUILD_CONFIG}.paths must be a JSON object containing "
            "project_root."
        )
    paths = value["paths"]
    if "project_root" not in paths or type(paths["project_root"]) is not str:
        raise TypeError(
            f"{DINO_OPS_BUILD_CONFIG}.paths.project_root must be an absolute string."
        )
    raw_root = paths["project_root"]
    if not raw_root or raw_root != raw_root.strip():
        raise ValueError(
            f"{DINO_OPS_BUILD_CONFIG}.paths.project_root must be non-empty and trimmed."
        )
    project_root = Path(raw_root)
    if not project_root.is_absolute():
        raise ValueError(
            f"{DINO_OPS_BUILD_CONFIG}.paths.project_root must be absolute; "
            f"got {raw_root!r}."
        )
    if project_root == Path(project_root.anchor):
        raise ValueError(
            f"{DINO_OPS_BUILD_CONFIG}.paths.project_root must not be the filesystem root."
        )
    if project_root.resolve(strict=False) != project_root:
        raise ValueError(
            f"{DINO_OPS_BUILD_CONFIG}.paths.project_root must already be resolved; "
            f"got {raw_root!r}."
        )
    operations_path = project_root / _OPERATIONS_RELATIVE_PATH
    if not operations_path.is_file():
        raise FileNotFoundError(
            "Canonical operation contract was not found below the configured "
            f"project_root: {operations_path}"
        )
    return project_root


def _load_operations(project_root: Path) -> ModuleType:
    """Import and authenticate the canonical contract from one explicit root."""
    for name, relative_path in _BOOTSTRAP_NAMESPACE_PATHS:
        if name in sys.modules:
            continue
        package_path = project_root / relative_path
        package = ModuleType(name)
        package.__package__ = name
        package.__dict__["__path__"] = [str(package_path)]
        package_spec = ModuleSpec(name, loader=None, is_package=True)
        package_spec.submodule_search_locations = [str(package_path)]
        package.__spec__ = package_spec
        sys.modules[name] = package

    root_text = str(project_root)
    added_to_path = root_text not in sys.path
    if added_to_path:
        sys.path.insert(0, root_text)
    try:
        module = importlib.import_module(_OPERATIONS_MODULE)
    finally:
        if added_to_path:
            sys.path.remove(root_text)

    loaded_file = module.__file__
    expected_file = project_root / _OPERATIONS_RELATIVE_PATH
    if loaded_file is None or Path(loaded_file).resolve(strict=False) != expected_file:
        raise RuntimeError(
            "Canonical operation contract resolved from an unexpected module path: "
            f"expected {expected_file}, got {loaded_file!r}."
        )
    return module


def _enabled_build_paths() -> Any | None:
    if not should_build_cuda_ops():
        return None
    raw = _required_build_json()
    project_root = _bootstrap_project_root(raw)
    operations = _load_operations(project_root)
    environment = operations.operation_environment()
    if environment.build_cuda_ops is not True:
        raise RuntimeError(
            "Canonical operation contract disagrees with the enabled packaging switch."
        )
    return environment.require_dino_build_config(repository_root=project_root)


def get_extensions() -> list[Any]:
    build_paths = _enabled_build_paths()
    if build_paths is None:
        return []
    build_target = selected_cuda_ops_build_target()
    _require_build_inputs(build_paths, build_target)

    try:
        from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
    except Exception as exc:  # pragma: no cover - depends on build environment
        raise RuntimeError("PyTorch is required to build tennis-lab CUDA ops") from exc

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME was not found. Set CUDA_HOME or install a CUDA-enabled "
            "PyTorch toolchain before building tennis-lab CUDA ops."
        )

    common_compile_args = {
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math"],
    }
    extensions = []
    if build_target == ALL_CUDA_OPS:
        extensions.append(
            CUDAExtension(
                name="src.utils.models.components.ops.time_local._C",
                sources=[
                    str(build_paths.time_local_bindings),
                    str(build_paths.time_local_kernels),
                ],
                extra_compile_args=common_compile_args,
            )
        )
    extensions.append(
        CUDAExtension(
            name="src.utils.models.components.ops.compressed_time_local._C",
            sources=[
                str(build_paths.compressed_time_local_bindings),
                str(build_paths.compressed_time_local_kernels),
            ],
            extra_compile_args=common_compile_args,
        )
    )
    if build_target == ALL_CUDA_OPS:
        dino_ops_src = _prepare_dino_ops_sources(
            build_paths.source,
            build_paths.destination,
        )
        extensions.append(
            CUDAExtension(
                name="MultiScaleDeformableAttention",
                sources=[
                    str(dino_ops_src / "vision.cpp"),
                    str(dino_ops_src / "cpu/ms_deform_attn_cpu.cpp"),
                    str(dino_ops_src / "cuda/ms_deform_attn_cuda.cu"),
                ],
                include_dirs=[str(dino_ops_src)],
                define_macros=[("WITH_CUDA", None)],
                extra_compile_args={
                    "cxx": [],
                    "nvcc": [
                        "-DCUDA_HAS_FP16=1",
                        "-D__CUDA_NO_HALF_OPERATORS__",
                        "-D__CUDA_NO_HALF_CONVERSIONS__",
                        "-D__CUDA_NO_HALF2_OPERATORS__",
                    ],
                },
            )
        )
    return extensions


def _require_build_inputs(build_paths: Any, build_target: str) -> None:
    """Require only the sources consumed by the selected extension build."""
    if build_target == ALL_CUDA_OPS:
        build_paths.require_inputs()
        return
    for path in (
        build_paths.compressed_time_local_bindings,
        build_paths.compressed_time_local_kernels,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"CUDA operation source file is missing: {path}")


def _prepare_dino_ops_sources(source: Path, destination: Path) -> Path:
    """Copy official DINO ops and apply the required modern-PyTorch dispatch fix."""
    cuda_source = source / "cuda/ms_deform_attn_cuda.cu"
    if not cuda_source.is_file():
        raise FileNotFoundError(
            "The configured DINO external asset is not initialized. "
            f"(missing: {cuda_source})"
        )

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    generated_cuda_source = destination / "cuda/ms_deform_attn_cuda.cu"
    contents = generated_cuda_source.read_text()
    replacement_count = contents.count(_OLD_DISPATCH)
    if replacement_count != 2:
        raise RuntimeError(
            "Unexpected DINO CUDA source: expected exactly two legacy dispatch "
            f"calls, found {replacement_count} in {cuda_source}"
        )
    generated_cuda_source.write_text(contents.replace(_OLD_DISPATCH, _NEW_DISPATCH))
    return destination


def get_cmdclass() -> dict[str, Any]:
    if _enabled_build_paths() is None:
        return {}

    try:
        from torch.utils.cpp_extension import BuildExtension
    except Exception as exc:  # pragma: no cover - depends on build environment
        raise RuntimeError("PyTorch is required to build tennis-lab CUDA ops") from exc
    return {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
