from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from setuptools import setup  # type: ignore[import-untyped]


def _load_ops_build_module() -> Any:
    build_path = (
        Path(__file__).parent
        / "src"
        / "utils"
        / "models"
        / "components"
        / "ops"
        / "build.py"
    )
    spec = importlib.util.spec_from_file_location("tennis_lab_ops_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load ops build module from {build_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ops_build = _load_ops_build_module()

setup(
    ext_modules=ops_build.get_extensions(),
    cmdclass=ops_build.get_cmdclass(),
)
