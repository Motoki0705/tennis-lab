"""Shared fixtures and configuration for e2e tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def tmp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-wide temporary output directory for training/inference outputs.

    Args:
        tmp_path_factory: pytest temp path factory

    Returns:
        Path: Temporary output directory

    """
    return tmp_path_factory.mktemp("outputs")


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-wide temporary data directory for test datasets.

    Args:
        tmp_path_factory: pytest temp path factory

    Returns:
        Path: Temporary data directory

    """
    return tmp_path_factory.mktemp("data")


@pytest.fixture
def run_hydra_script() -> Any:
    """Helper fixture to execute Hydra scripts with overrides.

    Returns:
        Callable that executes a Hydra script and returns subprocess result.

    """

    def _run(
        module_name: str,
        overrides: list[str] | None = None,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a Hydra script.

        Args:
            module_name: Python module name (e.g., "src.plcs.scripts.train")
            overrides: List of Hydra config overrides (e.g., ["run.gpus=0"])
            timeout: Timeout in seconds (default: 300s = 5 min)

        Returns:
            subprocess.CompletedProcess with stdout, stderr, and returncode

        """
        if overrides is None:
            overrides = []

        cmd = ["uv", "run", "python", "-m", module_name, *overrides]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result

    return _run
