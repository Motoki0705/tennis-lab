"""Loss, diagnostics, and post-fit checkpoint selection for PLCS residuals."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from pytorch_lightning.callbacks import ModelCheckpoint

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.training.residual_lightning_module import ResidualLightningModule
from src.tasks.plcs.training.residual_losses import residual_loss
from src.tasks.plcs.training.residual_metrics import paired_errors
from src.tasks.plcs.training.runner import PLCSTrainingRunner


def _recipe(tmp_path: Path | None = None):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        config = compose(config_name="train_triangulation_residual")
    config.model.hidden_dim = 32
    config.model.num_heads = 4
    config.model.ffn_dim = 64
    config.model.num_layers = 1
    if tmp_path is not None:
        config.paths.output_root = str(tmp_path)
    return config


def test_exact_supervised_residual_has_zero_loss_and_padding_is_masked() -> None:
    config = replace(
        validate_residual_config(_recipe()).loss,
        reprojection_weight=0.0,
        velocity_weight=0.0,
        bone_weight=0.0,
    )
    target = torch.randn(1, 6, 17, 3)
    root_init = torch.randn(1, 6, 3)
    relative_init = torch.randn(1, 6, 17, 3)
    target_root = target[:, :, [11, 12]].mean(2)
    output = {
        "root_residual": target_root - root_init,
        "relative_residual": target - target_root[:, :, None] - relative_init,
    }
    batch = {
        "features": torch.zeros(1, 2, 6, 367),
        "root_init": root_init,
        "relative_init": relative_init,
        "target_world": target,
        "frame_valid": torch.tensor([[True, True, True, False, False, False]]),
        "true_projection": torch.zeros(1, 2, 3, 4),
        "clean_uv": torch.zeros(1, 2, 6, 17, 2),
        "clean_visible": torch.zeros(1, 2, 6, 17, dtype=torch.bool),
        "fps": torch.tensor([30.0]),
    }
    loss, _, world = residual_loss(output, batch, config)
    assert loss < 1e-6
    torch.testing.assert_close(world, target, atol=1e-6, rtol=1e-6)
    corrupted = {key: value.clone() for key, value in output.items()}
    for value in corrupted.values():
        value[:, 3:] += 1000
    masked_loss, _, _ = residual_loss(corrupted, batch, config)
    torch.testing.assert_close(loss, masked_loss)


def test_tail_improvement_does_not_hide_population_regression() -> None:
    target = np.zeros((1, 100, 17, 3))
    initial = target.copy()
    initial[..., 0] = 0.1
    initial[:, -1, :, 0] = 10.0
    predicted = initial.copy()
    predicted[:, :-1, :, 0] += 0.01
    predicted[:, -1, :, 0] = 1.0
    summary = paired_errors(initial, predicted, target, np.ones((1, 100), bool))
    assert summary["predicted_m"]["mean"] < summary["initial_m"]["mean"]
    assert summary["predicted_m"]["median"] > summary["initial_m"]["median"]
    assert summary["improved_fraction"] == 0.01
    assert summary["worsened_fraction"] == 0.99


def test_runner_tests_the_validation_selected_checkpoint(tmp_path: Path) -> None:
    module = ResidualLightningModule(_recipe(tmp_path))
    module.residual_config.runtime.run.output_dir.mkdir(parents=True, exist_ok=True)
    callback = ModelCheckpoint(monitor="val/world_mpjpe_m")
    callback.best_model_path = str(tmp_path / "best.ckpt")
    callback.best_model_score = torch.tensor(0.1)
    Path(callback.best_model_path).touch()
    trainer = Mock()
    trainer.test.return_value = [{"test/world_mpjpe_m": 0.2}]
    datamodule = Mock()
    PLCSTrainingRunner().test_after_fit(trainer, module, datamodule, [callback])
    trainer.test.assert_called_once_with(
        module,
        datamodule=datamodule,
        ckpt_path=callback.best_model_path,
        weights_only=False,
    )
    assert (module.residual_config.runtime.run.output_dir / "evaluation.json").is_file()


def test_other_plcs_modules_keep_default_post_fit_test_behavior() -> None:
    trainer, module, data = Mock(), Mock(), Mock()
    PLCSTrainingRunner().test_after_fit(trainer, module, data, [])
    trainer.test.assert_called_once_with(module, datamodule=data)
