"""End-to-end checks for the knowledge-control registration entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.utils.training_queue import CEILING, RETENTION_NS

ROOT = Path(__file__).resolve().parents[3]
REGISTER_SCRIPT = ROOT / ".agents/skills/knowledge-control/scripts/kg_register.py"


def test_register_promotes_headline_and_diagnostic_metric_artifacts(
    tmp_path: Path,
) -> None:
    repro_dir = tmp_path / "repro" / "job-register-metrics"
    predictions_dir = repro_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    run_metadata = {
        "name": "register_metrics",
        "provider": "codex",
        "issue": "533",
        "command": "python -m train model=baseline loss=mse data=fixture",
        "resource": "half",
        "logical_gpu_slot": "1",
    }
    (repro_dir / "run.json").write_text(
        json.dumps(run_metadata),
        encoding="utf-8",
    )
    repro_script = "export TENNIS_GPU_RESOURCE=half\nexport TENNIS_GPU_SLOT=1\n"
    (repro_dir / "repro.sh").write_text(repro_script, encoding="utf-8")
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
            "--task",
            "plcs",
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
    assert json.loads((promoted_dir / "run.json").read_text(encoding="utf-8")) == (
        run_metadata
    )
    assert (promoted_dir / "repro.sh").read_text(encoding="utf-8") == repro_script
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

    node_text = (
        knowledge_dir / "nodes" / "plcs" / "000001-run-register-metrics.md"
    ).read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(node_text.split("---", maxsplit=2)[1])
    assert frontmatter["metrics"] == headline_metrics
    assert not set(diagnostic_metrics) & set(frontmatter["metrics"])


@pytest.mark.parametrize("expired", [False, True])
def test_named_registration_selects_latest_across_new_legacy_and_expired_history(
    tmp_path: Path,
    expired: bool,
) -> None:
    queue, knowledge = tmp_path / "queue", tmp_path / "knowledge"
    timestamp = 1_792_368_000_000_000_000
    name = "same_name"
    stems = [
        f"{timestamp - 1}_1_{name}",
        f"0-{CEILING - timestamp:019d}_2026-10-19T00-00-00-000000000Z_2_{name}",
        f"0-{CEILING - timestamp - 1:019d}_2026-10-19T00-00-00-000000001Z_3_{name}",
    ]
    for index, stem in enumerate(stems):
        job = queue / "cancelled" / f"{stem}.job"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text(
            f"# name: {name}\npython -m train model=baseline loss=mse data=fixture\n"
        )
        log = queue / "logs" / f"{stem}.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(f"| test/selected | {index} |\n")
        repro = queue / "repro" / stem
        (repro / "predictions").mkdir(parents=True)
        (repro / "run.json").write_text(
            json.dumps(
                dict(
                    name=name,
                    command="python -m train model=baseline",
                    provider="codex",
                    marker=index,
                )
            )
        )
        (repro / "predictions/metrics.json").write_text(
            json.dumps(dict(selected=index))
        )
    if expired:
        (queue / "cancelled" / f"{stems[-1]}.job").unlink()
        (queue / "logs" / f"{stems[-1]}.log").unlink()
        completion = queue / "control/completions" / f"{stems[-1]}.json"
        completion.parent.mkdir(parents=True)
        completion.write_text(
            json.dumps(
                dict(
                    version=1,
                    job=stems[-1] + ".job",
                    state="cancelled",
                    finished_at_ns=timestamp,
                    retention_started_at_ns=timestamp,
                    time_source="recorded",
                    expired_at_ns=timestamp + RETENTION_NS + 1,
                )
            )
        )
    environment = {
        **os.environ,
        "KNOWLEDGE_DIR": str(knowledge),
        "TRAINING_QUEUE_DIR": str(queue),
    }
    subprocess.run(
        [sys.executable, str(REGISTER_SCRIPT), name, "--task", "plcs"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    node = next((knowledge / "nodes").rglob("*.md"))
    values = yaml.safe_load(node.read_text().split("---", 2)[1])
    assert values["metrics"] == {"selected": 2}
    assert (
        json.loads((knowledge / "runs/run-same-name/run.json").read_text())["marker"]
        == 2
    )
    if expired:
        assert "log" not in values["artifacts"]
    else:
        assert values["artifacts"]["log"].endswith(stems[-1] + ".log")
    from_log = subprocess.run(
        [
            sys.executable,
            str(REGISTER_SCRIPT.with_name("kg_from_run.py")),
            name,
            "--task",
            "plcs",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    if expired:
        assert from_log.returncode == 2
        assert "queue history expired" in from_log.stderr
    else:
        assert from_log.returncode == 0, from_log.stderr
        values = yaml.safe_load(from_log.stdout.split("---", 2)[1])
        assert values["status"] == "failed"
        assert values["metrics"] == {"selected": 2}


def test_log_only_import_reads_new_name_without_metadata_header(tmp_path: Path) -> None:
    job = (
        tmp_path
        / "cancelled/0-8207631999999999999_2026-10-19T00-00-00-000000000Z_2_readable_name.job"
    )
    job.parent.mkdir()
    job.write_text("python -m train model=fixture\n")
    result = subprocess.run(
        [
            sys.executable,
            str(REGISTER_SCRIPT.with_name("kg_from_run.py")),
            "--job",
            str(job),
            "--task",
            "plcs",
        ],
        cwd=tmp_path,
        env={**os.environ, "TRAINING_QUEUE_DIR": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
    values = yaml.safe_load(result.stdout.split("---", 2)[1])
    assert values["title"] == "readable_name"
    assert values["status"] == "failed"
