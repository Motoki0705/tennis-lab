"""End-to-end checks for the knowledge-control registration entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
REGISTER_SCRIPT = ROOT / ".agents/skills/knowledge-control/scripts/kg_register.py"


def test_register_promotes_headline_and_diagnostic_metric_artifacts(
    tmp_path: Path,
) -> None:
    repro_dir = tmp_path / "repro" / "job-register-metrics"
    predictions_dir = repro_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    (repro_dir / "run.json").write_text(
        json.dumps(
            {
                "name": "register_metrics",
                "provider": "codex",
                "issue": "533",
                "command": "python -m train model=baseline loss=mse data=fixture",
            }
        ),
        encoding="utf-8",
    )
    (predictions_dir / "pred_test.npz").write_bytes(b"prediction-fixture")
    headline_metrics = {"position_error_m": 0.25, "angular_error_deg": 4.5}
    diagnostic_metrics = {"x_error_m": 0.1, "loss_position": 0.02}
    (predictions_dir / "metrics.json").write_text(
        json.dumps(headline_metrics),
        encoding="utf-8",
    )
    (predictions_dir / "diagnostic_metrics.json").write_text(
        json.dumps(diagnostic_metrics),
        encoding="utf-8",
    )

    knowledge_dir = tmp_path / "knowledge"
    subprocess.run(
        [
            sys.executable,
            str(REGISTER_SCRIPT),
            "--repro-dir",
            str(repro_dir),
            "--id",
            "run-register-metrics",
        ],
        cwd=ROOT,
        env={**os.environ, "KNOWLEDGE_DIR": str(knowledge_dir)},
        check=True,
        capture_output=True,
        text=True,
    )

    promoted_dir = knowledge_dir / "runs" / "run-register-metrics"
    assert (promoted_dir / "pred_test.npz").read_bytes() == b"prediction-fixture"
    assert json.loads((promoted_dir / "metrics.json").read_text(encoding="utf-8")) == (
        headline_metrics
    )
    assert (
        json.loads(
            (promoted_dir / "diagnostic_metrics.json").read_text(encoding="utf-8")
        )
        == diagnostic_metrics
    )

    node_text = (knowledge_dir / "nodes" / "run-register-metrics.md").read_text(
        encoding="utf-8"
    )
    frontmatter = yaml.safe_load(node_text.split("---", maxsplit=2)[1])
    assert frontmatter["metrics"] == headline_metrics
    assert not set(diagnostic_metrics) & set(frontmatter["metrics"])
