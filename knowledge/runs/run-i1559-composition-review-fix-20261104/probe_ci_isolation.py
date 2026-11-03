"""Seed the stale upstream functions state seen in CI, then reuse its unit node."""

import sys
from types import ModuleType

import pytest

from src.submodules.models.dino import architecture

name = f"{architecture._DINO_MODELS_PACKAGE}.dino.ops.functions.ms_deform_attn_func"
prior = ModuleType(name)
prior.MSDA = ModuleType("prior_extension")  # type: ignore[attr-defined]
sys.modules[name] = prior
exit_code = pytest.main([
    "-n", "0", "--durations=0", "--durations-min=0",
    "tests/unit/submodules/models/dino/test_extension.py::test_runtime_import_uses_selected_module_and_rejects_cached_mismatch",
])
assert sys.modules[name] is prior
raise SystemExit(exit_code)
