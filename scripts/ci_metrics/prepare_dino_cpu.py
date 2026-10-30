"""Build upstream DINO's import dependency for CPU matcher/loss acceptance.

Only the upstream pybind translation unit is needed: without WITH_CUDA its
attention entrypoints explicitly raise. This is not a CPU model forward backend.
No upstream source is patched or replaced.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from setuptools import setup  # type: ignore[import-untyped]
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("third_party/DINO"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.repository.resolve() / "models/dino/ops/src/vision.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"Initialize pinned third_party/DINO first: {source}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup(
        name="dino-cpu-loss-import",
        script_args=[
            "build_ext",
            "--build-lib",
            str(output),
            "--build-temp",
            str(output / "build"),
        ],
        ext_modules=[
            CppExtension(
                "MultiScaleDeformableAttention",
                [str(source)],
                extra_compile_args=["-O0", "-g0"],
            )
        ],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
    )


if __name__ == "__main__":
    main()
