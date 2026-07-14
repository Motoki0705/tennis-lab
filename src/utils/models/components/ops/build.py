from __future__ import annotations

import os
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "on"}


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
    import os
    dino_ops_src = os.path.abspath("src/submodules/vendor/dino/models/dino/ops/src")

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
                "src/submodules/vendor/dino/models/dino/ops/src/vision.cpp",
                "src/submodules/vendor/dino/models/dino/ops/src/cpu/ms_deform_attn_cpu.cpp",
                "src/submodules/vendor/dino/models/dino/ops/src/cuda/ms_deform_attn_cuda.cu",
            ],
            include_dirs=[dino_ops_src],
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


def get_cmdclass() -> dict[str, Any]:
    if not should_build_cuda_ops():
        return {}

    try:
        from torch.utils.cpp_extension import BuildExtension
    except Exception as exc:  # pragma: no cover - depends on build environment
        raise RuntimeError("PyTorch is required to build tennis-lab CUDA ops") from exc
    return {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
