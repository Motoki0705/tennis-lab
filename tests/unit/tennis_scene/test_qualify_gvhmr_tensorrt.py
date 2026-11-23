"""GVHMR qualification command path checks before model or report I/O."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.tennis_scene.scripts import qualify_gvhmr_tensorrt
from src.utils.configuration import PathContractError


def test_missing_progress_is_rejected_before_reading_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    report = tmp_path / "reports" / "qualification.json"
    progress = tmp_path / "previous_run" / "progress.json"
    monkeypatch.setattr(sys, "argv", [
        "qualify_gvhmr_tensorrt", str(bundle), str(report), "--ort-only", "--progress", str(progress),
    ])

    def unexpected_read(path: Path) -> None:
        pytest.fail(f"Opened the engine bundle before validating progress: {path}")

    monkeypatch.setattr(qualify_gvhmr_tensorrt.EngineBundle, "read", unexpected_read)
    with pytest.raises(PathContractError, match=r"input path does not exist for .*\.progress"):
        qualify_gvhmr_tensorrt.main()

    assert not report.parent.exists()
