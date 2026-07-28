"""Tests for the shell-free pinned NHT subprocess boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.pipeline import PipelineCommand
from src.synthetic_data_generation.rendering.nht.process import (
    NhtProcessBackend,
    NhtRuntime,
)

_COMMIT = "1" * 40


def _backend(tmp_path) -> NhtProcessBackend:
    repository = tmp_path / "nht"
    repository.mkdir()
    project = tmp_path / "project"
    (project / "src" / "synthetic_data_generation").mkdir(parents=True)
    return NhtProcessBackend(
        project_root=project,
        runtime=NhtRuntime(
            repository=repository,
            python=type(tmp_path)(sys.executable),
            expected_commit=_COMMIT,
        ),
    )


def test_nht_command_uses_pinned_python_and_module_without_shell(tmp_path) -> None:
    backend = _backend(tmp_path)
    command = PipelineCommand(
        stage="render",
        runtime="nht",
        module="src.synthetic_data_generation.dataset.blcs.rendering.nht",
        arguments=("--width", "320"),
    )

    argv = backend.command_for(command)

    assert argv[1:3] == ("-m", command.module)
    assert argv[-2:] == ("--width", "320")


def test_nht_runtime_preserves_virtualenv_interpreter_symlink(tmp_path) -> None:
    repository = tmp_path / "nht"
    repository.mkdir()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))

    runtime = NhtRuntime(
        repository=repository,
        python=venv_python,
        expected_commit=_COMMIT,
    )

    assert runtime.python == venv_python.absolute()
    assert runtime.python != venv_python.resolve()


def test_nht_runtime_rejects_commit_and_dirty_tree(monkeypatch, tmp_path) -> None:
    backend = _backend(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "2" * 40 + "\n",
    )
    with pytest.raises(RuntimeError, match="NHT commit differs"):
        backend.verify_runtime()

    outputs = iter((_COMMIT + "\n", " M tracked.py\n"))
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: next(outputs),
    )
    with pytest.raises(RuntimeError, match="tracked modifications"):
        backend.verify_runtime()
