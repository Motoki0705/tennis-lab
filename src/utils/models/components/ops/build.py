from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "on"}
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DINO_OPS_SOURCE = PROJECT_ROOT / "third_party/DINO/models/dino/ops/src"
DINO_OPS_BUILD_SOURCE = PROJECT_ROOT / "build/tennis_lab_dino_ops/src"
_OLD_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
_NEW_DISPATCH = "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"


def should_build_cuda_ops() -> bool:
    return os.environ.get("TENNIS_LAB_BUILD_CUDA_OPS", "").lower() in TRUE_VALUES


def get_extensions() -> list[Any]:
    if not should_build_cuda_ops():
        return []

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
    dino_ops_src = _prepare_dino_ops_sources(
        DINO_OPS_SOURCE,
        DINO_OPS_BUILD_SOURCE,
    )

    return [
        CUDAExtension(
            name="src.utils.models.components.ops.moe._C",
            sources=[
                "src/utils/models/components/ops/moe/bindings.cpp",
                "src/utils/models/components/ops/moe/kernels.cu",
            ],
            extra_compile_args=common_compile_args,
        ),
        CUDAExtension(
            name="src.utils.models.components.ops.time_local._C",
            sources=[
                "src/utils/models/components/ops/time_local/bindings.cpp",
                "src/utils/models/components/ops/time_local/kernels.cu",
            ],
            extra_compile_args=common_compile_args,
        ),
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
        ),
    ]


def _prepare_dino_ops_sources(source: Path, destination: Path) -> Path:
    """Copy official DINO ops and apply the required modern-PyTorch dispatch fix."""
    cuda_source = source / "cuda/ms_deform_attn_cuda.cu"
    if not cuda_source.is_file():
        raise FileNotFoundError(
            "DINO git submodule is not initialized. Run: "
            "git submodule update --init third_party/DINO "
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
    if not should_build_cuda_ops():
        return {}

    try:
        from torch.utils.cpp_extension import BuildExtension
    except Exception as exc:  # pragma: no cover - depends on build environment
        raise RuntimeError("PyTorch is required to build tennis-lab CUDA ops") from exc
    return {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
