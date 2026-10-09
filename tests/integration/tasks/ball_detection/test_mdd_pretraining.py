import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from src.tasks.ball_detection.models.mdd_pretrain import DeepMDDQueryDetector
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    transfer_encoder,
)
from tests.integration.tasks.ball_detection.test_coordinate_training import (
    preparation as preparation_fixture,
)
from tests.unit.tasks.ball_detection.models.test_mdd_pretrain import small_config


def test_pretraining_cli_saves_evaluation_gif_and_transferable_encoder(tmp_path: Path) -> None:
    prepared, plan = preparation_fixture.__wrapped__(tmp_path)
    model_config = tmp_path / "pretraining.yaml"
    model_config.write_text(yaml.safe_dump(dict(name="mdd_dpt_pretrain", **asdict(small_config()))))
    output = tmp_path / "pretrained"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.pretrain_mdd_dpt",
        "--manifest", plan["datasets"]["mdd_only"]["path"], "--model-config", str(model_config),
        "--output", str(output), "--epochs", "1", "--windows-per-epoch", "3", "--warmup-updates", "0",
        "--learning-rate", ".001", "--device", "cpu", "--precision", "fp32", "--jpeg-decoder", "opencv",
        "--num-workers", "0", "--compile-mode", "off", "--preview-clips", "1"],
        env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    completed = json.loads((output / "COMPLETED.json").read_text())
    assert completed["global_step"] == 3 and completed["automatic_next_stage"] is False
    report = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
    assert set(report["scopes"]) == {"full", "common"}
    assert report["scopes"]["full"]["macro_heatmap_loss"] > 0
    assert list((output / "previews").rglob("*.gif"))
    target = DeepMDDQueryDetector(small_config())
    receipt = transfer_encoder(output / "epoch-000.pt", target)
    assert receipt["global_step"] == 3
    saved = torch.load(output / "epoch-000.pt", weights_only=True)
    assert saved["recipe"]["test_usage"] == "none"


def test_diagnostic_prefix_resumes_with_the_original_schedule_and_weights(tmp_path: Path) -> None:
    _, plan = preparation_fixture.__wrapped__(tmp_path)
    model_config = tmp_path / "pretraining.yaml"
    model_config.write_text(yaml.safe_dump(dict(name="mdd_dpt_pretrain", **asdict(small_config()))))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.pretrain_mdd_dpt",
        "--manifest", plan["datasets"]["mdd_only"]["path"], "--model-config", str(model_config),
        "--epochs", "2", "--windows-per-epoch", "3", "--warmup-updates", "1", "--learning-rate", ".001",
        "--device", "cpu", "--precision", "fp32", "--jpeg-decoder", "opencv", "--num-workers", "0",
        "--compile-mode", "off", "--preview-clips", "0", "--log-every", "1"]

    def run(output: Path, extra: list[str]) -> None:
        result = subprocess.run([*command, "--output", str(output), *extra],
            env=env, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr

    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(full, [])
    run(resumed, ["--stop-after-epoch", "0"])
    stop = json.loads((resumed / "DIAGNOSTIC_STOP.json").read_text())
    assert stop["global_step"] == 3 and stop["total_updates"] == 6
    assert stop["complete"] is False and not (resumed / "COMPLETED.json").exists()
    run(resumed, ["--resume", str(resumed / "epoch-000.pt")])
    actual = torch.load(resumed / "epoch-001.pt", weights_only=True)
    expected = torch.load(full / "epoch-001.pt", weights_only=True)
    assert actual["recipe"] == expected["recipe"]
    assert actual["global_step"] == expected["global_step"] == 6
    assert torch.equal(actual["torch_rng"], expected["torch_rng"])
    assert all(torch.equal(actual["state_dict"][key], value) for key, value in expected["state_dict"].items())
    logs = [[json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()] for path in (full, resumed)]
    assert [(r["learning_rate"], r["train_loss"]) for r in logs[0]] == [(r["learning_rate"], r["train_loss"]) for r in logs[1]]
    assert (resumed / "COMPLETED.json").is_file()
