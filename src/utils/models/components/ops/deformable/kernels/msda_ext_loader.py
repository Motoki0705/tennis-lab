"""Extension loading logic for MSDA CUDA kernels."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _configure_build_env() -> None:
    """Set conservative defaults for extension build/cache environment."""
    default_ext_dir = Path("outputs/torch_extensions").resolve()
    default_ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(default_ext_dir))

    cpu_count = os.cpu_count() or 1
    os.environ.setdefault("MAX_JOBS", str(min(cpu_count, 8)))

    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return

    arch_values: set[str] = set()
    for device_idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(device_idx)
        arch_values.add(f"{major}.{minor}")
    if arch_values:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(arch_values))


def try_load_msda_extension(*, already_attempted: bool, cached_ext: object | None) -> tuple[object | None, bool]:
    """Load CUDA extension once and return updated `(ext, attempted)` state."""
    if already_attempted:
        return cached_ext, True

    attempted = True
    if os.environ.get("MSDA_FORCE_FALLBACK", "0") == "1":
        return None, attempted
    if not torch.cuda.is_available():
        return None, attempted
    _configure_build_env()

    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
    sources = [
        str(csrc_dir / "binding.cpp"),
        str(csrc_dir / "deformable_cuda.cpp"),
        str(csrc_dir / "deformable_cuda_kernel.cu"),
    ]

    try:
        ext = load(
            name="msda_ops",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        ext = None

    return ext, attempted
