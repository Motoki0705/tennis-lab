"""End-to-end tests for the generic Hydra dataset pipeline entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_dataset_pipeline_cli_runs_from_configured_paths(tmp_path: Path) -> None:
    source = tmp_path / "third_party/nht/data"
    artifacts = tmp_path / "third_party/nht/artifacts/synthetic-data"
    outputs = tmp_path / "outputs/synthetic_data_generation"
    dataset = tmp_path / "data/synthetic_data_generation"
    source.mkdir(parents=True)
    (source / "alignment-observations.json").write_text(
        json.dumps({"residuals": [999_999.0]}),
        encoding="utf-8",
    )
    (source / "prepared.bin").write_bytes(b"\x00")
    (source / "reference.bin").write_bytes(b"\xff")
    (source / "render-jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "poor-sample",
                        "input": "prepared.bin",
                        "output": "renders/poor-sample.bin",
                        "reference": "reference.bin",
                        "arguments": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    hydra_dir = tmp_path / "hydra"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.dataset.run_pipeline",
            f"roots.project_root={tmp_path}",
            "roots.data_root=data",
            "roots.checkpoint_root=ckpt",
            "roots.artifact_root=third_party/nht/artifacts",
            "roots.output_root=outputs",
            "roots.cache_root=.cache",
            "roots.external_asset_root=third_party",
            "execute=true",
            f"hydra.run.dir={hydra_dir}",
        ],
        check=True,
        cwd=Path.cwd(),
        env=environment,
    )

    path_manifest = json.loads((outputs / "path-manifest.json").read_text())
    alignment = json.loads((artifacts / "alignment-metrics.json").read_text())
    quality = json.loads((artifacts / "quality-metrics.json").read_text())
    assert path_manifest["paths"]["source_root"] == str(source)
    assert alignment["root_mean_square_error"] == 999_999.0
    assert quality["mean_absolute_byte_error"] == 1.0
    assert (dataset / "renders/poor-sample.bin").is_file()
    assert (outputs / "pipeline-summary.html").is_file()


def test_dataset_pipeline_rejects_job_escape_before_pipeline_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "third_party/nht/data"
    source.mkdir(parents=True)
    (source / "alignment-observations.json").write_text(
        json.dumps({"residuals": [0.0]}),
        encoding="utf-8",
    )
    (source / "prepared.bin").write_bytes(b"input")
    (source / "render-jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "escape",
                        "input": "prepared.bin",
                        "output": "../outside.bin",
                        "reference": None,
                        "arguments": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd())

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.dataset.run_pipeline",
            f"roots.project_root={tmp_path}",
            "roots.data_root=data",
            "roots.checkpoint_root=ckpt",
            "roots.artifact_root=third_party/nht/artifacts",
            "roots.output_root=outputs",
            "roots.cache_root=.cache",
            "roots.external_asset_root=third_party",
            "execute=true",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        check=False,
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "outputs/path-manifest.json").exists()
    assert not (tmp_path / "third_party/nht/artifacts/synthetic-data").exists()
    assert not (tmp_path / "data/synthetic_data_generation").exists()
