"""Process-exit tests for the tennis-scene dataset generation entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from src.utils.paths import PROJECT_ROOT

_EXIT_CODE_ENV = "TENNIS_LAB_GENERATE_DATASET_TEST_EXIT_CODE"
_ENTRYPOINT_HARNESS = f"""
import os
import runpy
from unittest.mock import patch


def fake_hydra_main(*args, **kwargs):
    del args, kwargs

    def decorate(function):
        del function
        return lambda: int(os.environ[{_EXIT_CODE_ENV!r}])

    return decorate


with (
    patch("src.utils.hydra.hydra_main", fake_hydra_main),
    patch("src.utils.hydra.register_boundary_validator", lambda *args: None),
):
    runpy.run_module(
        "src.tennis_scene.scripts.generate_dataset",
        run_name="__main__",
    )
"""


@pytest.mark.parametrize("expected_exit_code", [0, 1])
def test_generate_dataset_cli_propagates_main_result(
    expected_exit_code: int,
) -> None:
    environment = {
        **os.environ,
        _EXIT_CODE_ENV: str(expected_exit_code),
    }

    completed = subprocess.run(
        [sys.executable, "-c", _ENTRYPOINT_HARNESS],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == expected_exit_code, completed.stderr
