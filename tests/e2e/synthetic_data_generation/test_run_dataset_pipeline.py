"""End-to-end smoke test for the Hydra dataset pipeline entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_dataset_pipeline_cli_publishes_immutable_plan(tmp_path: Path) -> None:
    plan = tmp_path / "court-plan.json"
    hydra_dir = tmp_path / "hydra"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic_data_generation.scripts.dataset.run_pipeline",
            "domain=court",
            f"plan_path={plan}",
            f"hydra.run.dir={hydra_dir}",
        ],
        check=True,
        cwd=Path.cwd(),
        env=environment,
    )

    payload = json.loads(plan.read_text())
    assert payload["dataset"] == "court"
    assert payload["selected_algorithms"]["camera_sampling"] == "inward_orbit"
    assert [item["stage"] for item in payload["commands"]] == ["runtime_probe"]
