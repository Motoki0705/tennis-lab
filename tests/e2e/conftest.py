from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

import pytest
import torch

F = TypeVar("F", bound=Callable[..., object])


def pytest_configure(config: pytest.Config) -> None:
    """Register the cuda marker with pytest."""
    config.addinivalue_line(
        "markers",
        "cuda: mark test as requiring CUDA/GPU (skipped if GPU unavailable)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked with @pytest.mark.cuda if CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


def typed_fixture(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Provide a typed pytest fixture decorator for mypy."""
    return cast(Callable[[F], F], pytest.fixture(*args, **kwargs))


def typed_mark(mark: pytest.MarkDecorator) -> Callable[[F], F]:
    """Provide a typed pytest mark decorator for mypy."""
    return cast(Callable[[F], F], mark)


@typed_fixture(scope="session")
def schema_validators() -> ModuleType:
    """Provide schema validation utilities module."""
    from tests.e2e import validation

    return cast(ModuleType, validation)


@typed_fixture(scope="session")
def tmp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temp output directory for e2e runs."""
    return Path(tmp_path_factory.mktemp("output"))


@typed_fixture(scope="session")
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temp data directory for e2e runs."""
    return Path(tmp_path_factory.mktemp("data"))


@typed_fixture(scope="session")
def run_hydra_script(
    tmp_output_dir: Path,
    tmp_data_dir: Path,
) -> Callable[[str, Sequence[str] | None, Path | None], subprocess.CompletedProcess[str]]:
    """Run a Hydra-enabled module with standard temp dirs."""

    def _run(
        module: str,
        extra_args: Sequence[str] | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        args: Iterable[str] = (
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            module,
            f"run.output_dir={tmp_output_dir}",
            f"run.data_dir={tmp_data_dir}",
        )
        command = list(args)
        if extra_args:
            command.extend(extra_args)
        return subprocess.run(
            command,
            check=False,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
        )

    return _run
