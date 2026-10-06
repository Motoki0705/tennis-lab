"""CPU subprocess fixtures for the bounded SfM experiment recorder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("outcome", ["success", "missing", "failure"])
def test_recorder_keeps_measured_output_and_failure_status(
    tmp_path: Path, outcome: str
) -> None:
    metrics = tmp_path / "observed.json"
    repro = tmp_path / "repro"
    script = (
        "from pathlib import Path; "
        f"Path({str(metrics)!r}).write_text('{{\"registration_ratio\": 0.5}}')"
        if outcome == "success"
        else "raise SystemExit(7)"
        if outcome == "failure"
        else "pass"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.sfm_comparison.run_command",
            "--campaign",
            str(tmp_path / "campaign"),
            "--run-id",
            outcome,
            "--stage",
            "sfm",
            "--seconds",
            "10",
            "--metrics-source",
            str(metrics),
            "--required-path",
            str(metrics),
            "--no-memory-sampling",
            "--",
            sys.executable,
            "-c",
            script,
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "TENNIS_RUN_ID": "cpu-fixture",
            "TENNIS_REPRO_DIR": str(repro),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    saved = json.loads((repro / "execution.json").read_text())
    assert (result.returncode == 0) == (outcome == "success")
    assert saved["status"] == ("done" if outcome == "success" else "failed")
    assert saved["metrics"] == (
        {"registration_ratio": 0.5} if outcome == "success" else {}
    )
    assert saved["device_peak_memory_mib"] is None
    assert saved["elapsed_seconds"] >= 0
    flat = json.loads((repro / "predictions/metrics.json").read_text())
    assert all(isinstance(value, (int, float)) for value in flat.values())
    assert "device_peak_memory_mib" not in flat  # Missing samples are not zero.
    if outcome == "success":
        assert flat["sfm/registration_ratio"] == 0.5


def test_recorder_rejects_direct_execution(tmp_path: Path) -> None:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("TENNIS_")
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.sfm_comparison.run_command",
            "--campaign",
            str(tmp_path),
            "--run-id",
            "direct",
            "--stage",
            "sfm",
            "--seconds",
            "10",
            "--required-path",
            str(tmp_path),
            "--",
            "true",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "training-queue" in result.stderr
    assert not (tmp_path / "runs").exists()
