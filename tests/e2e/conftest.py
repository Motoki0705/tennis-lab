from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

import pytest

F = TypeVar("F", bound=Callable[..., object])


def typed_fixture(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Provide a typed pytest fixture decorator for mypy."""
    return cast(Callable[[F], F], pytest.fixture(*args, **kwargs))


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
