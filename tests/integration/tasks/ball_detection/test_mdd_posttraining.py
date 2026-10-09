import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml

from src.tasks.ball_detection.training.posttraining.augmentation import (
    AugmentationConfig,
)
from tests.integration.tasks.ball_detection.test_coordinate_training import (
    preparation as preparation_fixture,
)
from tests.unit.tasks.ball_detection.models.test_mdd_pretrain import small_config


def test_pretrained_cnn_freeze_then_joint_training_and_fixed_stress_reports(tmp_path: Path) -> None:
    _, plan = preparation_fixture.__wrapped__(tmp_path)
    manifest = plan["datasets"]["mdd_only"]["path"]
    cfg = tmp_path / "model.yaml"
    cfg.write_text(yaml.safe_dump(dict(name="mdd_dpt_pretrain", **asdict(small_config()))))
    aug = tmp_path / "augmentation.yaml"
    aug.write_text(yaml.safe_dump(asdict(replace(AugmentationConfig.load(Path(__file__).resolve().parents[4] / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"), camera_probability=1., occlusion_probability=1.))))
    pre, post = tmp_path / "pre", tmp_path / "post"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    base = ["--manifest", manifest, "--device", "cpu", "--precision", "fp32", "--jpeg-decoder", "opencv",
            "--num-workers", "0", "--compile-mode", "off", "--warmup-updates", "0", "--windows-per-epoch", "3", "--log-every", "1"]
    first = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.pretrain_mdd_dpt", *base,
        "--model-config", str(cfg), "--output", str(pre), "--epochs", "1", "--learning-rate", ".001", "--preview-clips", "0"],
        env=env, capture_output=True, text=True, timeout=180)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.posttrain_mdd_query", *base,
        "--pretraining-run", str(pre), "--augmentation-config", str(aug), "--output", str(post),
        "--epochs", "2", "--freeze-epochs", "1", "--learning-rate", ".001", "--stress-evaluation", "--preview-clips", "1"],
        env=env, capture_output=True, text=True, timeout=240)
    assert second.returncode == 0, second.stdout + second.stderr
    initial = torch.load(pre / "epoch-000.pt", weights_only=True)["state_dict"]
    frozen = torch.load(post / "epoch-000.pt", weights_only=True)["state_dict"]
    joint = torch.load(post / "epoch-001.pt", weights_only=True)["state_dict"]
    names = [key for key in initial if key.startswith("encoder.")]
    assert all(torch.equal(initial[key], frozen[key]) for key in names)
    assert any(not torch.equal(initial[key], joint[key]) for key in names)
    logs = [json.loads(line) for line in (post / "train.jsonl").read_text().splitlines()]
    assert all(r["encoder_frozen"] and r["encoder_learning_rate"] == 0 for r in logs if r["epoch"] == 0)
    assert all(not r["encoder_frozen"] and r["encoder_learning_rate"] > 0 for r in logs if r["epoch"] == 1)
    assert set(json.loads((post / "stress.json").read_text())) == {"camera", "occlusion", "combined"}
    assert json.loads((post / "COMPLETED.json").read_text())["stage"] == "query_posttraining"
    assert list((post / "previews").rglob("*.gif"))
