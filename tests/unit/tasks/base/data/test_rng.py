"""Unit tests for deterministic scene data RNG ownership."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from src.tasks.base.data.rng import (
    derive_seed,
    require_run_seed,
    validate_run_seed,
)
from src.utils.configuration import MissingConfigurationKeyError

pytestmark = pytest.mark.unit


def test_run_seed_validation_is_exact_and_fail_closed() -> None:
    assert validate_run_seed(0) == 0
    assert validate_run_seed(2**32 - 1) == 2**32 - 1
    with pytest.raises(TypeError, match="must be an int"):
        validate_run_seed(True)
    with pytest.raises(ValueError, match="between"):
        validate_run_seed(-1)
    with pytest.raises(ValueError, match="between"):
        validate_run_seed(2**32)


def test_required_run_seed_has_no_default() -> None:
    assert require_run_seed({"run": {"seed": 753}}) == 753
    with pytest.raises(MissingConfigurationKeyError, match="configuration.run"):
        require_run_seed({})
    with pytest.raises(MissingConfigurationKeyError, match="run.seed"):
        require_run_seed({"run": {}})


def test_seed_derivation_is_domain_separated() -> None:
    assert derive_seed(42, "dataset", "train") == derive_seed(
        42, "dataset", "train"
    )
    assert derive_seed(42, "dataset", "train") != derive_seed(
        42, "loader", "train"
    )
    assert derive_seed(42, "ab", "c") != derive_seed(42, "a", "bc")
    assert derive_seed(42, "1") != derive_seed(42, 1)


def test_seed_derivation_is_independent_of_python_hash_seed() -> None:
    script = (
        "from src.tasks.base.data.rng import derive_seed; "
        "print(derive_seed(42, 'dataset', 'train'))"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                text=True,
                env=environment,
            ).strip()
        )
    assert outputs[0] == outputs[1]
