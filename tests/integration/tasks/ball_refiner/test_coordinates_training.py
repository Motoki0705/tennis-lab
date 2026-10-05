"""Real CPU BLCS -> shared storage -> train/checkpoint/test/inference roundtrip."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.ball_refiner.coordinates.config import CorruptionConfig, parse_section
from src.tasks.ball_refiner.coordinates.data import SharedDataset, prepare
from src.tasks.ball_refiner.coordinates.detection import heatmaps_to_coordinates
from src.tasks.ball_refiner.coordinates.evaluation import evaluate
from src.tasks.ball_refiner.coordinates.generation import (
    CameraSampling,
    generate_dataset,
)
from src.tasks.ball_refiner.coordinates.inference import load_checkpoint
from src.tasks.ball_refiner.coordinates.reprojection import reproject_dataset
from src.tasks.ball_refiner.coordinates.training import run_training
from src.tasks.ball_refiner.scripts.predict_coordinates import predict_file
from src.utils.paths import PROJECT_ROOT

CONFIGS = str(PROJECT_ROOT / "src/tasks/ball_refiner/configs")


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared-coordinate-data")
    with initialize_config_dir(version_base=None, config_dir=CONFIGS):
        cfg = compose(config_name="generate_coordinates", overrides=[
            f"paths.data_root={root}", "generation.train_rallies=2", "generation.val_rallies=1", "generation.test_rallies=1", "generation.workers=1",
        ])
    path = generate_dataset(cfg)
    with pytest.raises(FileExistsError):
        generate_dataset(cfg)
    return root, path


def test_reprojection_reuses_identical_3d_events_and_splits(shared, tmp_path):
    _, source = shared
    target = reproject_dataset(source, tmp_path / "new", CameraSampling(4, 1280, 720, 3, 12, 60, 120, 2, 12, 0.5, 2, 1024), seed=85)
    first, second = SharedDataset(source), SharedDataset(target)
    for before, after in zip(first.rallies, second.rallies, strict=True):
        assert before.split == after.split and before.name == after.name
        np.testing.assert_array_equal(before.xyz, after.xyz)
        np.testing.assert_array_equal(before.events, after.events)
        np.testing.assert_array_equal(before.time, after.time)
        assert after.visible.all()
        assert not np.array_equal(before.uv, after.uv)


@pytest.mark.parametrize("dimensions,architecture,gan", [(2, "regression", 0.002), (3, "regression", 0.002), (3, "flow", 0)])
def test_shared_training_roundtrip(shared, tmp_path, dimensions, architecture, gan):
    torch.set_num_threads(1)
    root, dataset_path = shared
    data = SharedDataset(dataset_path)
    assert len(list((dataset_path / "rallies").glob("*.npz"))) == 4
    split_ids = {name: {r.name for r in data.split(name)} for name in ("train", "val", "test")}
    assert not split_ids["train"] & (split_ids["val"] | split_ids["test"])
    assert not split_ids["val"] & split_ids["test"]
    with initialize_config_dir(version_base=None, config_dir=CONFIGS):
        cfg = compose(config_name="train_coordinates", overrides=[
            f"paths.data_root={root}", f"paths.output_root={tmp_path}", "run.output_dir=ball_refiner/train/integration/cpu",
            "run.device=cpu", "training.steps=2", "training.batch_size=2", "training.evaluate_every=2", "training.log_every=1", "training.gan_warmup_steps=0",
            f"training.gan_weight={gan}", f"model.dimensions={dimensions}", f"model.architecture={architecture}",
            "model.width=16", "model.layers=1", "model.heads=2", "model.window_length=32", "model.flow_steps=3",
        ])
    output = run_training(cfg)
    with pytest.raises(FileExistsError):
        run_training(cfg)
    checkpoint = output / "logs/version_0/checkpoints/best.ckpt"
    model, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    assert metadata["manifest_sha256"] == data.manifest_hash
    config = replace(parse_section(CorruptionConfig, dict(cfg.corruption)), event_probability=cfg.data.evaluation_event_probability)
    test = prepare(data.split("test"), dimensions, config, cfg.data.evaluation_seed)
    _, repeated = evaluate(model, test, torch.device("cpu"), seed=cfg.data.evaluation_seed, batch_size=2)
    with np.load(output / "predictions/pred_test.npz") as saved:
        np.testing.assert_array_equal(saved["prediction"], repeated["prediction"])
    rally = test[0]
    coordinates = rally.corrupted.uv_px if dimensions == 2 else rally.corrupted.xyz_m[None]
    missing = rally.corrupted.missing_2d if dimensions == 2 else rally.corrupted.missing_3d[None]
    source = tmp_path / "input.npz"
    extras = {"image_size_wh": np.array([1280, 720])} if dimensions == 2 else {}
    np.savez(source, coordinates=coordinates, missing=missing, fps=np.asarray(60), **extras)
    destination = tmp_path / "prediction.npz"
    predict_file(checkpoint, source, destination, device="cpu", seed=42, batch_size=2)
    with np.load(destination) as predicted:
        assert predicted["coordinates"].shape == coordinates.shape
        assert np.isfinite(predicted["coordinates"]).all()
        np.testing.assert_array_equal(predicted["input_missing"], missing)
    with pytest.raises(FileExistsError):
        predict_file(checkpoint, source, destination, device="cpu", seed=42, batch_size=2)


def test_heatmap_conversion_uses_geometry_and_explicit_mask_only():
    heatmaps = torch.zeros(1, 3, 9, 17)
    heatmaps[0, :, 4, 8] = torch.tensor([0.1, 0.5, 0.9])
    missing = torch.tensor([[False, True, False]])
    points = heatmaps_to_coordinates(heatmaps, missing, image_size_wh=(1280, 720))
    torch.testing.assert_close(points[0, 0], torch.tensor([639.5, 359.5]))
    torch.testing.assert_close(points[0, 0], points[0, 2])
    assert points[0, 1].count_nonzero() == 0
