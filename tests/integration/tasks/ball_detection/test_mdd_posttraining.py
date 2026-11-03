"""Encoder handoff, stress reports and the historical query resume boundary."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml

from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
from src.tasks.ball_detection.evaluation import coordinate_cli
from src.tasks.ball_detection.training.coordinates import finetune
from tests.integration.tasks.ball_detection.lifecycle_contracts import (
    exercise_mdd_resume,
)
from tests.integration.tasks.ball_detection.mdd_training_contracts import (
    encoder_arguments,
    invoke_main,
    omit_explicit_channels,
    pretrained_run,
    read_json,
    weights,
)
from tests.integration.tasks.ball_detection.model_contracts import family
from tests.unit.tasks.ball_detection.models.mdd.test_cnn import small_config

pytestmark = pytest.mark.integration


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


def test_encoder_handoff_freeze_joint_and_fixed_stress_reports(tmp_path: Path) -> None:
    inputs, pre = pretrained_run(tmp_path)
    output = tmp_path / "post"
    arguments = encoder_arguments(inputs.manifest, pre, output)
    default = Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
    augmentation = tmp_path / "augmentation.yaml"
    augmentation.write_text(yaml.safe_dump(asdict(replace(
        AugmentationConfig.load(default), camera_probability=1., occlusion_probability=1.,
    ))))
    invoke_main(finetune.main, [
        *arguments, "--augmentation-config", str(augmentation),
        "--stress-evaluation", "--preview-clips", "1",
    ])
    parent = weights(pre / "epoch-000.pt")
    frozen, joint = (weights(output / f"epoch-{epoch:03d}.pt") for epoch in (0, 1))
    names = [key for key in parent["state_dict"] if key.startswith("encoder.")]
    assert all(torch.equal(parent["state_dict"][key], frozen["state_dict"][key]) for key in names)
    assert any(not torch.equal(parent["state_dict"][key], joint["state_dict"][key]) for key in names)
    assert joint["recipe"]["parent"]["mode"] == "encoder"
    assert joint["recipe"]["parent"]["ancestry"]["data"] == parent["recipe"]["data_identity"]
    assert joint["recipe"]["data_identity"] == parent["recipe"]["data_identity"]
    logs = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [(r["epoch"], r["global_step"], r["encoder_frozen"]) for r in logs] == [(0, 1, True), (1, 2, False)]
    assert logs[0]["encoder_learning_rate"] == 0 < logs[1]["encoder_learning_rate"]
    assert set(read_json(output / "stress.json")) == {"camera", "occlusion", "combined"}
    assert read_json(output / "COMPLETED.json")["stage"] == "query_posttraining"
    assert joint["recipe"]["runtime"]["input_verification"] == "lazy"
    assert list((output / "previews").rglob("*.gif"))

    evaluation = tmp_path / "evaluation.json"
    invoke_main(coordinate_cli.main, [
        "--checkpoint", str(output / "epoch-001.pt"), "--manifest", str(inputs.manifest),
        "--output", str(evaluation), "--split", "val", "--device", "cpu",
        "--artifact-root", str(tmp_path), "--output-root", str(tmp_path), "--cpu-threads", "1",
    ])
    assert read_json(evaluation)["scopes"] == joint["validation"]["scopes"]


def test_historical_query_restores_then_matches_the_next_update(tmp_path: Path) -> None:
    exercise_mdd_resume(
        family("mdd_dpt_pretrain", "deep_query", name="historical_query"),
        tmp_path, epochs=2, mutate=omit_explicit_channels,
    )
