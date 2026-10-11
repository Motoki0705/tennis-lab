import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml

from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
)
from tests.integration.tasks.ball_detection.test_coordinate_training import (
    preparation as preparation_fixture,
)
from tests.unit.tasks.ball_detection.models.mdd.test_cnn import small_config


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; invoke through shared training queue")
def test_shared_epoch_bf16_compile_freeze_and_unfreeze(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
    from src.tasks.ball_detection.training.coordinates.loop import train_epoch
    from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
    from src.tasks.ball_detection.training.schedules import FineTuneSchedule

    device = torch.device("cuda")
    runtime = CoordinateRuntime(precision="bf16", compile_mode="default")
    runtime.configure(device)
    torch.manual_seed(42)
    cfg = replace(small_config(), encoder_variant="convnext_v2", temporal_mixing="factorized")
    model = DeepMDDQueryDetector(cfg).to(device)
    optimizer = torch.optim.AdamW([
        dict(params=list(model.encoder.parameters()), role="encoder"),
        dict(params=[p for n, p in model.named_parameters() if not n.startswith("encoder.")], role="decoder"),
    ], lr=.001)
    initial = {name: p.detach().clone() for name, p in model.encoder.named_parameters()}
    samples = [dict(rgb=torch.randint(256, (32, 3, 64, 96), dtype=torch.uint8),
                    timestamps=torch.arange(32).float() / 30, uv=torch.full((32, 2), .4),
                    position_valid=torch.ones(32, dtype=torch.bool), clip_id="gpu-smoke")]
    loader = DataLoader(samples, batch_size=1)
    runtime.configure_model(model)
    schedule = FineTuneSchedule(.001, 2, 0, 1, .1)
    for epoch in range(2):
        result = train_epoch(model, optimizer, loader, device, runtime, epoch=epoch, global_step=epoch,
                             window_budget=1, output=tmp_path, log_every=1, schedule=schedule)
        assert result.global_step == epoch + 1
        assert torch.isfinite(torch.tensor(result.mean_loss))
        if epoch == 0:
            assert all(torch.equal(initial[name], p) for name, p in model.encoder.named_parameters())
            assert all(p.grad is None for p in model.encoder.parameters())
    assert any(not torch.equal(initial[name], p) for name, p in model.encoder.named_parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())


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
    first = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.train", "--stage", "dpt_pretrain", *base,
        "--model-config", str(cfg), "--output", str(pre), "--epochs", "1", "--learning-rate", ".001", "--preview-clips", "0"],
        env=env, capture_output=True, text=True, timeout=180)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.train", "--stage", "coordinates", "--initialize", "encoder", *base,
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

    evaluation = tmp_path / "post-evaluation.json"
    evaluated = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.evaluate",
        "--mode", "coordinates", "--checkpoint", str(post / "epoch-001.pt"),
        "--manifest", manifest, "--output", str(evaluation), "--split", "val", "--device", "cpu",
        "--artifact-root", str(tmp_path), "--output-root", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=180)
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    latest = json.loads((post / "metrics.jsonl").read_text().splitlines()[-1])
    assert json.loads(evaluation.read_text())["scopes"] == latest["scopes"]

    # Recreate an interrupted run from its completed frozen epoch, then resume.
    import shutil

    resumed = tmp_path / "resumed"
    shutil.copytree(post, resumed)
    for name in ("epoch-001.pt", "COMPLETED.json", "stress.json"):
        (resumed / name).unlink()
    for name in ("metrics.jsonl", "train.jsonl"):
        rows = [json.loads(line) for line in (resumed / name).read_text().splitlines()]
        (resumed / name).write_text(''.join(json.dumps(row) + '\n' for row in rows if row['epoch'] == 0))
    result = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.train",
        "--stage", "coordinates", "--initialize", "encoder", *base,
        "--pretraining-run", str(pre), "--augmentation-config", str(aug), "--output", str(resumed),
        "--resume", str(resumed / "epoch-000.pt"), "--epochs", "2", "--freeze-epochs", "1",
        "--learning-rate", ".001", "--stress-evaluation", "--preview-clips", "1"],
        env=env, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    replayed = torch.load(resumed / "epoch-001.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(replayed[name], value) for name, value in joint.items())
    assert json.loads((resumed / "stress.json").read_text()) == json.loads((post / "stress.json").read_text())
