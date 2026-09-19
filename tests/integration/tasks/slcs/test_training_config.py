"""SLCS training configuration integration tests."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tasks.slcs.training.lightning_module import SLCSLightningModule

_CONFIG_DIR = Path(__file__).parents[4] / "src" / "tasks" / "slcs" / "configs"


def test_lightning_schedule_uses_trainer_max_epochs() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=["model=small", "training.trainer.max_epochs=7"],
        )

    module = SLCSLightningModule(config)

    assert module.max_epochs == 7


def test_small_model_defaults_to_original_all_shared_trunk() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train", overrides=["model=small"])

    model = SLCSLightningModule(config).model

    assert model.num_shared_layers == 2
    assert model.num_position_layers == 0
    assert model.num_rotation_layers == 0


def test_legacy_config_checkpoint_loads_with_ablations_disabled(tmp_path: Path) -> None:
    from pytorch_lightning import __version__ as lightning_version

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train", overrides=["model=small"])
    with open_dict(config.loss):
        del config.loss.ball_velocity_weight
        del config.loss.ball_velocity_scale_mps
    with open_dict(config.model):
        del config.model.missing_ball_court_context
    module = SLCSLightningModule(config)
    path = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {"config": config},
            "pytorch-lightning_version": lightning_version,
        },
        path,
    )
    restored = SLCSLightningModule.load_from_checkpoint(path, weights_only=False)
    assert restored.loss_fn.config.ball_velocity_weight == 0.0
    assert restored.loss_fn.config.ball_velocity_scale_mps == 1.0
    assert restored.model.missing_ball_context is None
    restored.eval()
    module.eval()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    spec = restored.model_adapter.spec
    batch = {
        "player_kp": torch.zeros(1, spec.num_players, 2, 17, 2),
        "player_kp_vis": torch.ones(1, spec.num_players, 2, 17),
        "player_valid": torch.ones(1, spec.num_players, 2, dtype=torch.bool),
        "ball_uv": torch.zeros(1, 2, 2),
        "ball_vis": torch.ones(1, 2, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, spec.num_court_kp, 2),
        "court_vis": torch.ones(1, 2, spec.num_court_kp),
        "padding_mask": torch.zeros(1, 2, dtype=torch.bool),
        "dino_tokens": torch.zeros(1, 1, spec.dino_num_tokens, spec.dino_embed_dim),
        "dino_frame_idx": torch.zeros(1, 1, dtype=torch.int64),
        "dino_padding_mask": torch.zeros(1, 1, dtype=torch.bool),
    }
    with torch.no_grad():
        expected = module.forward_batch(batch)
        actual = restored.forward_batch(batch)
    torch.testing.assert_close(
        actual.ball_position, expected.ball_position, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.player_position, expected.player_position, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.player_rotation, expected.player_rotation, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "override",
    [
        "loss.ball_velocity_weight=-1",
        "loss.ball_velocity_scale_mps=0",
        "loss.ball_velocity_scale_mps=.nan",
        "model.missing_ball_court_context=1",
        "model.missing_ball_court_context=invalid",
    ],
)
def test_ablation_config_rejects_bad_values(override: str) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train", overrides=[override])
    with pytest.raises(ValueError):
        SLCSTrainingRuntimeConfig.from_config(config)


def test_court_context_profile_reaches_runtime_model() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_real_rgb_missing_ball_court")
    module = SLCSLightningModule(config)
    assert module.model.missing_ball_context is not None
    assert torch.count_nonzero(module.model.missing_ball_context.weight) == 0
