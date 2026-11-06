"""Reproduce CI's pre-existing DINO child-module cache before a fake import."""

import sys
from types import ModuleType

import pytest


def main() -> int:
    name = "_tennis_lab_third_party_dino_models.dino.ops.functions.ms_deform_attn_func"
    prior = ModuleType(name)
    prior.MSDA = ModuleType("previous_extension")  # type: ignore[attr-defined]
    sys.modules[name] = prior
    print(f"Injected prior upstream module: {name}", flush=True)
    node = (
        "tests/unit/submodules/models/dino/test_extension.py::"
        "test_runtime_import_uses_selected_module_and_rejects_cached_mismatch"
    )
    return int(pytest.main([
        "-o", "addopts=", "--import-mode=importlib", "-n", "0", "-q",
        "--durations=0", "--durations-min=0", node, *sys.argv[1:],
    ]))


if __name__ == "__main__":
    raise SystemExit(main())
