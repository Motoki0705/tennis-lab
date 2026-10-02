"""The supported single-object runtime, including checkpoint and GAN wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.base.model_io import bind_model_io
from src.tasks.blcs.configuration import build_path_resolver, validate_training_boundary
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io.training import compose_blcs_training
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.training.composition import build_plcs_lightning_module


def _config(task: str, root: Path, *, gan: bool = False) -> DictConfig:
    overrides = [
        "model.hidden_dim=16",
        "model.num_heads=2",
        "model.ffn_dim=32",
        "model.rope_dim=8",
        "model.num_layers=1",
        "model.dropout=0.0",
        "data.seq_len_range=[3,3]",
        "data.num_views_range=[2,2]",
        "data.num_workers=0",
        "data.batch_size=1",
        "training.compile.enabled=false",
        f"paths.checkpoint_root={root}",
    ]
    if task == "blcs":
        overrides.extend(
            [
                "model.camera_layers_per_stage=[1]",
                "model.time_layers_per_stage=[1]",
                "model.time_global_stage_mask=[true]",
            ]
        )
    if gan:
        overrides.extend(
            [
                "training=gan_small",
                "training.gan.discriminator.hidden_dim=16",
                "training.gan.discriminator.num_layers=1",
                "training.gan.discriminator.num_heads=2",
                "training.gan.discriminator.ffn_dim=32",
                "training.gan.discriminator.rope_dim=8",
            ]
        )
    with initialize_config_dir(
        config_dir=str(Path("src/tasks", task, "configs").resolve()), version_base="1.3"
    ):
        return compose(config_name="train", overrides=overrides)


def _module(task: str, config: DictConfig) -> Any:
    if task == "blcs":
        return compose_blcs_training(config, generator_config=None).lightning_module
    return build_plcs_lightning_module(config)


def _observations(task: str) -> dict[str, torch.Tensor]:
    values = {
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 3, dtype=torch.bool),
    }
    if task == "blcs":
        values.update(
            ball_uv=torch.rand(1, 2, 3, 2),
            ball_vis=torch.ones(1, 2, 3, dtype=torch.bool),
        )
    else:
        values.update(
            human_kp=torch.rand(1, 2, 3, 17, 2),
            human_vis=torch.ones(1, 2, 3, 17, dtype=torch.bool),
        )
    return values


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_axial_checkpoint_roundtrip_preserves_predictions(
    task: str, tmp_path: Path
) -> None:
    config = _config(task, tmp_path)
    module = _module(task, config).eval()
    observations = _observations(task)
    with torch.no_grad():
        expected = module.model_io.run(observations).position
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": {"config": config},
        "pytorch-lightning_version": pl.__version__,
    }
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "axial.ckpt"
    torch.save(checkpoint, path)
    resolver = build_path_resolver(config)
    predictor_type = BLCSPredictor if task == "blcs" else PLCSPredictor
    predictor = predictor_type.load_from_checkpoint(
        path, resolver=resolver, device="cpu"
    )
    binding = bind_model_io(predictor.model, predictor.io_adapter)
    with torch.no_grad():
        actual = binding.run(observations).position
    assert actual.shape == (1, 3, 3)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_gan_keeps_a_trainable_discriminator(task: str, tmp_path: Path) -> None:
    module = _module(task, _config(task, tmp_path, gan=True))
    assert module.discriminator is not None
    assert any(
        parameter.requires_grad for parameter in module.discriminator.parameters()
    )
    assert module.automatic_optimization is False


@pytest.mark.parametrize("task", ["blcs", "plcs"])
@pytest.mark.parametrize(
    "suffix", ["track_query", "multiview_axial_reference", "triangulation_residual"]
)
def test_retired_models_are_rejected_before_runtime_construction(
    task: str, suffix: str, tmp_path: Path
) -> None:
    config = _config(task, tmp_path)
    config.model.name = f"{task}_{suffix}"
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        _module(task, config)


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_camera_view_generation_contract_is_rejected(task: str, tmp_path: Path) -> None:
    config = _config(task, tmp_path)
    config.court_keypoints.selector = "camera_view_v2"
    with pytest.raises(ValueError, match="physical_v1"):
        if task == "blcs":
            validate_training_boundary(config)
        else:
            PLCSTrainingConfig.from_config(config)
