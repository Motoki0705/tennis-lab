"""Real CPU BLCS -> shared storage -> train/checkpoint/test/inference roundtrip."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.ball_refiner_3d.configuration.core import parse_section
from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.configuration.generation import CameraSampling
from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.ball_refiner_3d.data.preprocessing import prepare
from src.tasks.ball_refiner_3d.evaluation.evaluator import evaluate
from src.tasks.ball_refiner_3d.generate_dataset.generator import generate_dataset
from src.tasks.ball_refiner_3d.generate_dataset.reprojection import reproject_dataset
from src.tasks.ball_refiner_3d.inference.files import predict_file
from src.tasks.ball_refiner_3d.model_io.checkpoint import load_checkpoint
from src.tasks.ball_refiner_3d.training.runner import run_training
from src.utils.paths import PROJECT_ROOT

CONFIGS = str(PROJECT_ROOT / "src/tasks/ball_refiner_3d/configs")


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared-coordinate-data")
    with initialize_config_dir(version_base=None, config_dir=CONFIGS):
        cfg = compose(
            config_name="generate_dataset",
            overrides=[
                f"paths.data_root={root}",
                "generation.train_rallies=2",
                "generation.val_rallies=1",
                "generation.test_rallies=1",
                "generation.workers=1",
            ],
        )
    path = generate_dataset(cfg)
    with pytest.raises(FileExistsError):
        generate_dataset(cfg)
    return root, path


def test_reprojection_reuses_identical_3d_events_and_splits(shared, tmp_path):
    _, source = shared
    target = reproject_dataset(
        source,
        tmp_path / "new",
        CameraSampling(4, 1280, 720, 3, 12, 60, 120, 2, 12, 0.5, 2, 1024),
        seed=85,
    )
    first, second = SharedDataset(source), SharedDataset(target)
    for before, after in zip(first.rallies, second.rallies, strict=True):
        assert before.split == after.split and before.name == after.name
        np.testing.assert_array_equal(before.xyz, after.xyz)
        np.testing.assert_array_equal(before.events, after.events)
        np.testing.assert_array_equal(before.time, after.time)
        assert after.visible.all()
        assert not np.array_equal(before.uv, after.uv)


@pytest.mark.parametrize(
    "dimensions,architecture,gan,gan_only",
    [
        (3, "regression", False, False),
        (3, "regression", True, False),
        (3, "flow", False, False),
        (3, "regression", True, True),
    ],
)
def test_shared_training_roundtrip(
    shared, tmp_path, monkeypatch, dimensions, architecture, gan, gan_only
):
    torch.set_num_threads(1)
    root, dataset_path = shared
    data = SharedDataset(dataset_path)
    assert len(list((dataset_path / "rallies").glob("*.npz"))) == 4
    split_ids = {
        name: {r.name for r in data.split(name)} for name in ("train", "val", "test")
    }
    assert not split_ids["train"] & (split_ids["val"] | split_ids["test"])
    assert not split_ids["val"] & split_ids["test"]
    schedule_overrides = (
        [
            "loss.reconstruction.enabled=true",
            "loss.reconstruction.final_weight=0",
            "loss.reconstruction.start_step=1",
            "loss.reconstruction.decay_steps=2",
        ]
        if gan_only
        else []
    )
    with initialize_config_dir(version_base=None, config_dir=CONFIGS):
        cfg = compose(
            config_name="train",
            overrides=[
                *schedule_overrides,
                f"paths.data_root={root}",
                f"paths.output_root={tmp_path}",
                "run.output_dir=ball_refiner/train/integration/cpu",
                "run.gpus=0",
                "training.steps=4",
                "training.batch_size=2",
                "training.evaluate_every=2",
                "training.log_every=1",
                "training.gan.transition.start_step=1",
                "training.gan.schedule_enabled=true",
                "training.gan.warmup_steps=2",
                "training.gan.discriminator.num_layers=1",
                f"training.gan.enabled={str(bool(gan)).lower()}",
                f"model.dimensions={dimensions}",
                f"model.architecture={architecture}",
                "model.ffn_dim=64",
                "model.rope_dim=8",
                "model.width=16",
                "model.layers=1",
                "model.heads=2",
                "model.window_length=32",
                "model.flow_steps=3",
            ],
        )
    output = run_training(cfg)
    with pytest.raises(FileExistsError):
        run_training(cfg)
    rows = [
        json.loads(line)
        for line in (output / "learning_curve.jsonl").read_text().splitlines()
    ]
    target_weight = 1.0
    assert [row["gan_weight_current"] for row in rows] == (
        [0.0, target_weight / 2, target_weight, target_weight] if gan else [0.0] * 4
    )
    assert [row["reconstruction_weight_current"] for row in rows] == (
        [1.0, 0.5, 0.0, 0.0] if gan_only else [1.0] * 4
    )
    for row in rows:
        assert row["weighted_reconstruction"] == pytest.approx(
            row["reconstruction"] * row["reconstruction_weight_current"]
        )
        assert row["total"] == pytest.approx(
            row["weighted_reconstruction"]
            + row["weighted_gan"]
            + row["weighted_event"],
            rel=1e-5,
        )
    last = torch.load(
        output / "logs/version_0/checkpoints/last.ckpt", weights_only=True
    )
    if gan:
        assert {
            int(state["step"])
            for state in last["optimizer_states"][1]["state"].values()
        } == {3}
        assert last["gan_weight"] == target_weight
    assert last["reconstruction_weight"] == (0.0 if gan_only else 1.0)
    if gan_only:
        assert all(
            row["total"] == pytest.approx(row["weighted_gan"] + row["weighted_event"])
            for row in rows[2:]
        )
    checkpoint = output / "logs/version_0/checkpoints/best.ckpt"
    model, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    assert metadata["manifest_sha256"] == data.manifest_hash
    from src.tasks.ball_refiner_3d.visualization.dataset_review.checkpoints import (
        describe,
    )

    entry = describe(
        checkpoint, "trained", data.manifest_hash, data.fps, tmp_path / "review-cache"
    )
    assert entry.info["compatible"], entry.info["reason"]
    assert entry.info["method"] == (
        "flow" if architecture == "flow" else "gan" if gan else "regression"
    )
    assert (
        entry.info["evaluation_profile"]["augmentation_seed"]
        == cfg.data.evaluation_seed
    )
    config = replace(
        parse_section(CorruptionConfig, dict(cfg.augmentation)),
        event_probability=cfg.data.evaluation_event_probability,
    )
    test = prepare(
        data.split("test"),
        dimensions,
        config,
        cfg.data.evaluation_seed,
        event_sigma_frames=cfg.loss.event.sigma_frames,
    )
    _, repeated = evaluate(
        model, test, torch.device("cpu"), seed=cfg.data.evaluation_seed, batch_size=2
    )
    with np.load(output / "predictions/pred_test.npz") as saved:
        np.testing.assert_array_equal(saved["prediction"], repeated["prediction"])
        np.testing.assert_array_equal(
            saved["event_probability"], repeated["event_probability"]
        )
    final_model, _ = load_checkpoint(
        output / "logs/version_0/checkpoints/last.ckpt", torch.device("cpu")
    )
    _, repeated_last = evaluate(
        final_model,
        test,
        torch.device("cpu"),
        seed=cfg.data.evaluation_seed,
        batch_size=2,
    )
    with (
        np.load(output / "predictions_last/pred_test.npz") as final,
        np.load(output / "predictions/pred_test.npz") as saved,
    ):
        np.testing.assert_array_equal(final["prediction"], repeated_last["prediction"])
        for key in (
            "target",
            "input",
            "missing",
            "event",
            "event_target",
            "rally_id",
            "view_id",
            "frame_id",
        ):
            np.testing.assert_array_equal(final[key], saved[key])
    first_report = json.loads(
        (output / "predictions/diagnostic_metrics.json").read_text()
    )
    final_report = json.loads(
        (output / "predictions_last/diagnostic_metrics.json").read_text()
    )
    assert (
        first_report["evaluation_input_sha256"]
        == final_report["evaluation_input_sha256"]
    )
    assert (
        final_report["checkpoint_kind"] == "last"
        and final_report["checkpoint_step"] == 4
    )
    assert first_report["checkpoint_step"] == metadata["step"]
    for directory in ("predictions", "predictions_last"):
        metrics = json.loads((output / directory / "metrics.json").read_text())
        assert all(
            type(value) in (int, float) for value in metrics.values()
        )  # knowledge importer contract
    rally = test[0]
    coordinates = rally.corrupted.xyz_m[None]
    missing = rally.corrupted.missing_3d[None]
    source = tmp_path / "input.npz"
    np.savez(source, coordinates=coordinates, missing=missing, fps=np.asarray(60))
    destination = tmp_path / "prediction.npz"
    predict_file(checkpoint, source, destination, device="cpu", seed=42, batch_size=2)
    with np.load(destination) as predicted:
        assert predicted["coordinates"].shape == coordinates.shape
        assert np.isfinite(predicted["coordinates"]).all()
        assert predicted["event_probability"].shape == missing.shape
        assert (
            (predicted["event_probability"] >= 0)
            & (predicted["event_probability"] <= 1)
        ).all()
        np.testing.assert_array_equal(predicted["input_missing"], missing)
    with pytest.raises(FileExistsError):
        predict_file(
            checkpoint, source, destination, device="cpu", seed=42, batch_size=2
        )


@pytest.mark.parametrize(
    "architecture,gan", [("regression", False), ("flow", False), ("regression", True)]
)
def test_native_resume_matches_uninterrupted_partial_final_epoch(
    shared, tmp_path, architecture, gan
):
    import pytorch_lightning as pl

    from src.tasks.ball_refiner_3d.training.runner import RefinerTrainingRunner

    class DeliberateStop(RuntimeError):
        pass

    class StopAfterBlock(pl.Callback):
        def on_train_epoch_start(self, trainer, module):
            if trainer.current_epoch == 1:
                raise DeliberateStop(
                    "simulate interruption after saved validation block"
                )

    root, _ = shared
    with initialize_config_dir(version_base=None, config_dir=CONFIGS):
        config = compose(
            config_name="train",
            overrides=[
                f"paths.data_root={root}",
                f"paths.output_root={tmp_path}",
                "run.output_dir=interrupted",
                "run.gpus=0",
                "run.test_after_fit=false",
                "training.steps=5",
                "training.evaluate_every=2",
                "training.log_every=3",
                "training.batch_size=2",
                f"model={architecture}",
                "model.width=16",
                "model.layers=1",
                "model.heads=2",
                "model.rope_dim=8",
                "model.ffn_dim=64",
                "model.window_length=32",
                "model.flow_steps=3",
                "model.dropout=0.1",
                f"training.gan.enabled={str(gan).lower()}",
                "training.gan.schedule_enabled=true",
                "training.gan.transition.start_step=1",
                "training.gan.warmup_steps=2",
                "training.gan.discriminator.num_layers=1",
            ],
        )

    class InterruptibleRunner(RefinerTrainingRunner):
        def callbacks_extra(self, *_):
            return [StopAfterBlock()]

    runner = InterruptibleRunner()
    with pytest.raises(DeliberateStop):
        runner.run(config)
    output = tmp_path / "interrupted"
    checkpoint = output / "logs/version_0/checkpoints/last.ckpt"
    assert torch.load(checkpoint, weights_only=True)["step"] == 2
    config.paths.checkpoint_root = str(tmp_path)
    config.run.resume = "interrupted/logs/version_0/checkpoints/last.ckpt"
    run_training(config)
    resumed = torch.load(checkpoint, weights_only=True)
    config.run.resume = None
    config.run.output_dir = "uninterrupted"
    full = run_training(config)
    uninterrupted = torch.load(
        full / "logs/version_0/checkpoints/last.ckpt", weights_only=True
    )
    assert resumed["step"] == uninterrupted["step"] == 5
    assert resumed["epoch"] == uninterrupted["epoch"] == 2
    for key, expected in uninterrupted["state_dict"].items():
        torch.testing.assert_close(resumed["state_dict"][key], expected, rtol=0, atol=0)
    for key in ("flow_rng", "torch_rng"):
        torch.testing.assert_close(resumed[key], uninterrupted[key], rtol=0, atol=0)
    actual = [
        json.loads(row)
        for row in (output / "learning_curve.jsonl").read_text().splitlines()
    ]
    expected = [
        json.loads(row)
        for row in (full / "learning_curve.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in actual] == [3, 5]
    assert [{k: v for k, v in r.items() if k != "seconds"} for r in actual] == [
        {k: v for k, v in r.items() if k != "seconds"} for r in expected
    ]
