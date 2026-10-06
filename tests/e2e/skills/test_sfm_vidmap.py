"""A failed mapper must not produce an evaluated success artifact."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from experiments.sfm_comparison import run_vidmap


def arguments(output: Path) -> list[str]:
    return [
        "run_vidmap",
        "--images",
        "images",
        "--manifest",
        "manifest.json",
        "--output",
        str(output),
        "--evaluator-root",
        "nht",
    ]


def test_refuses_direct_gpu_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path / "output"))
    with pytest.raises(SystemExit) as result:
        run_vidmap.main()
    assert result.value.code == 2


def test_failed_mapper_cannot_reach_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("TENNIS_RUN_ID", "cpu-test-fixture")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path / "repro"))
    monkeypatch.setattr(sys, "argv", arguments(output))

    def failed_mapper(command: list[str], *, check: bool) -> None:
        if not check:
            raise AssertionError("Mapper errors must propagate")
        raise subprocess.CalledProcessError(7, command)

    def unexpected_evaluation(*args: object) -> None:
        raise AssertionError("Failed mapping must not be evaluated")

    monkeypatch.setattr(run_vidmap.subprocess, "run", failed_mapper)
    monkeypatch.setattr(run_vidmap, "evaluate", unexpected_evaluation)
    with pytest.raises(subprocess.CalledProcessError):
        run_vidmap.main()
    assert not (output / "phase-times.json").exists()
