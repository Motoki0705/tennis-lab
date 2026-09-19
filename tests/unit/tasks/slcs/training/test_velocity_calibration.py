"""Train-only scope, reproducibility, and output-gradient calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.splits import load_split_assignments
from src.tasks.slcs.training import velocity_calibration as calibration


def test_toy_gradient_ratio_is_exact() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = calibration.calibrate_gradient_ratio(
        2 * pred.sum(), 5 * pred.sum(), pred, ratio=0.1
    )
    assert result["ball_velocity_weight"] == pytest.approx(0.04)
    assert result["weighted_velocity_gradient_norm"] / result[
        "supervised_gradient_norm"
    ] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "failure",
    [
        "zero_supervised",
        "zero_velocity",
        "nan_gradient",
        "nan_loss",
        "vector",
        "ratio",
        "detached",
    ],
)
def test_invalid_gradient_inputs_fail(failure: str) -> None:
    pred = torch.tensor([0.0, 1.0], requires_grad=True)
    supervised = pred.sum()
    velocity = 2 * pred.sum()
    ratio = 0.1
    if failure == "zero_supervised":
        supervised = 0 * supervised
    elif failure == "zero_velocity":
        velocity = 0 * velocity
    elif failure == "nan_gradient":
        velocity = pred.sqrt().sum()  # finite loss; infinite gradient at zero
    elif failure == "nan_loss":
        velocity = velocity * float("nan")
    elif failure == "vector":
        velocity = pred
    elif failure == "ratio":
        ratio = 0
    else:
        pred = pred.detach()
    with pytest.raises(ValueError):
        calibration.calibrate_gradient_ratio(supervised, velocity, pred, ratio=ratio)


@pytest.fixture
def saved_run(
    tmp_path: Path, synthetic_dataset: SLCSDataIndex, synthetic_split_file: Path
) -> tuple[Path, Path]:
    root = tmp_path / "outputs"
    run = root / "slcs/train/fixture/s42-001"
    run.mkdir(parents=True)
    config_dir = Path(__file__).parents[5] / "src/tasks/slcs/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_real_rgb",
            overrides=[
                f"paths.data_root={synthetic_dataset.root.parent}",
                f"paths.output_root={root}",
                f"data.dataset_root={synthetic_dataset.root.name}",
                f"data.split_file={synthetic_dataset.root.name}/{synthetic_split_file.name}",
                "data.window_size=8",
                "data.train_stride=8",
                "data.eval_stride=8",
                "data.dino.image_height=48",
                "data.dino.image_width=64",
                "data.dino.embed_dim=8",
                "model.hidden_dim=32",
                "model.num_shared_layers=1",
                "model.ffn_dim=64",
                "model.rope_dim=8",
                "model.dino_patch_downsample_factor=1",
            ],
        )
    OmegaConf.save(config, run / "config.yaml", resolve=True)
    return root, run


def test_cpu_small_model_smoke_is_train_only_and_reproducible(
    saved_run: tuple[Path, Path],
    synthetic_dataset: SLCSDataIndex,
    synthetic_split_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.slcs.data import dataset as dataset_module

    assignments = load_split_assignments(synthetic_split_file, synthetic_dataset)
    original_load = dataset_module.load_clip_arrays
    original_dataset = calibration.SLCSWindowDataset
    split_calls: list[str] = []
    label_reads: list[str] = []

    def guarded_load(manifest: Any, **kwargs: Any) -> Any:
        assert assignments[manifest.video_id] == "train"
        label_reads.append(manifest.video_id)
        return original_load(manifest, **kwargs)

    def guarded_dataset(**kwargs: Any) -> Any:
        assert kwargs["split"] == "train"
        assert kwargs["augment"] is True
        assert kwargs["config"].require_dino is True
        assert kwargs["config"].augmentation is not None
        split_calls.append(kwargs["split"])
        return original_dataset(**kwargs)

    def reject_checkpoint(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Calibration must not load a checkpoint")

    monkeypatch.setattr(dataset_module, "load_clip_arrays", guarded_load)
    monkeypatch.setattr(calibration, "SLCSWindowDataset", guarded_dataset)
    monkeypatch.setattr(torch, "load", reject_checkpoint)
    root, _ = saved_run
    reports = []
    for name in ("one", "two"):
        path = calibration.calibrate_training_run(
            output_root=root,
            training_run="slcs/train/fixture/s42-001",
            output=f"slcs/analyze/calibration/{name}",
            velocity_scale_mps=11.25,
            batch_size=2,
            seed=42,
        )
        reports.append(json.loads(path.read_text()))
    assert reports[0] == reports[1]
    report = reports[0]
    assert split_calls == ["train", "train"] and label_reads
    assert report["model_training"] is True and report["checkpoint_loaded"] is False
    assert len(set(report["selected_indices"])) == 2
    assert all(
        assignments[w["video_id"]] == "train" for w in report["selected_windows"]
    )
    assert report["valid_pair_count"] >= report["positive_weight_pair_count"] > 0
    assert report["weighted_velocity_gradient_norm"] / report[
        "supervised_gradient_norm"
    ] == pytest.approx(0.1)
    with pytest.raises(FileExistsError):
        calibration.calibrate_training_run(
            output_root=root,
            training_run="slcs/train/fixture/s42-001",
            output="slcs/analyze/calibration/one",
            velocity_scale_mps=11.25,
        )


@pytest.mark.parametrize("field,value", [("overfit", True), ("on_incomplete", "skip")])
def test_scope_config_rejected_before_dataset(
    saved_run: tuple[Path, Path],
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run = saved_run
    config = OmegaConf.load(run / "config.yaml")
    config.data[field] = value
    OmegaConf.save(config, run / "config.yaml")
    monkeypatch.setattr(
        calibration,
        "SLCSWindowDataset",
        lambda **_: pytest.fail("scope rejected before dataset"),
    )
    with pytest.raises(ValueError, match="overfit and skip-incomplete"):
        calibration.calibrate_training_run(
            output_root=root,
            training_run="slcs/train/fixture/s42-001",
            output="slcs/analyze/calibration/bad",
            velocity_scale_mps=1.0,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"seed": -1},
        {"velocity_scale_mps": 0.0},
        {"gradient_ratio": float("nan")},
        {"output_root": Path("relative")},
        {"training_run": "/absolute"},
        {"output": "slcs/analyze/../escape"},
        {"output": "slcs/evaluate/a/b"},
        {"device": "mps"},
    ],
)
def test_invalid_arguments_fail_before_io(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    args: dict[str, Any] = dict(
        output_root=tmp_path,
        training_run="slcs/train/fixture/s42-001",
        output="slcs/analyze/calibration/one",
        velocity_scale_mps=1.0,
    )
    args.update(overrides)
    with pytest.raises(ValueError):
        calibration.calibrate_training_run(**args)


def test_cuda_requires_queue_before_device_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TENNIS_RUN_ID", raising=False)
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    with pytest.raises(ValueError, match="training queue"):
        calibration.calibrate_training_run(
            output_root=tmp_path,
            training_run="slcs/train/fixture/s42-001",
            output="slcs/analyze/calibration/one",
            device="cuda",
            velocity_scale_mps=1.0,
        )


def test_output_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    (root / "slcs/analyze").mkdir(parents=True)
    (root / "slcs/analyze/escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="within output_root"):
        calibration.calibrate_training_run(
            output_root=root,
            training_run="slcs/train/fixture/s42-001",
            output="slcs/analyze/escape/run",
            velocity_scale_mps=1.0,
        )
