"""Command-level tests for the alignment pipeline Hydra entry point."""

from __future__ import annotations

import subprocess
import sys


def test_pipeline_cli_composes_b00_job_without_executing_gpu_stages() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.run_alignment_pipeline",
            "--cfg",
            "job",
            "jobs=b00",
            "stages=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pipeline_id: alignment-batch-v1" in completed.stdout
    assert "provider_bundle: data/tennis/3dgs_scenes/b00-default-v1" in completed.stdout
    assert "stages: all" in completed.stdout


def test_pipeline_cli_accepts_design_csv_stage_syntax() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.run_alignment_pipeline",
            "--cfg",
            "job",
            "jobs=b00",
            "stages=fit,calibrate,finalize",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "stages: fit,calibrate,finalize" in completed.stdout
