"""Weights-only ball FT must accept 2D checkpoints and reject partial loading."""

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.ball_detection.training.runner import BallDetectionTrainingRunner
from src.tasks.ball_detection.training.staged_runner import (
    StagedBallDetectionTrainingRunner,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs"


def _config(tmp_path: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_meiji_mixed",
            overrides=[
                f"paths.checkpoint_root={tmp_path}",
                "run.init_weights=init.ckpt",
                "model.dims=[2,4,8,16]",
                "model.depth=1",
            ],
        )


@pytest.mark.parametrize(
    "runner_type", [BallDetectionTrainingRunner, StagedBallDetectionTrainingRunner],
)
def test_init_loads_all_legacy_2d_weights_without_court_metadata(
    tmp_path: Path, runner_type: type[BallDetectionTrainingRunner],
) -> None:
    config = _config(tmp_path)
    original = BallDetectionLightningModule(config)
    state = original.state_dict()
    # A deployed 2D checkpoint predates the 3D court normalization contract.
    torch.save(
        {"state_dict": state, "epoch": 13, "optimizer_states": [{"irrelevant": 1}]},
        tmp_path / "init.ckpt",
    )
    restored = BallDetectionLightningModule(config)
    assert any(not torch.equal(state[k], v) for k, v in restored.state_dict().items())
    runner = runner_type()
    runner.maybe_load_init_weights(runner.validate_runtime_config(config), restored)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert restored.current_epoch == 0


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_init_rejects_incomplete_or_incompatible_state(
    tmp_path: Path, corruption: str,
) -> None:
    config = _config(tmp_path)
    module = BallDetectionLightningModule(config)
    state = module.state_dict()
    key = next(iter(state))
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        state["unrecognized.weight"] = torch.ones(1)
    else:
        state[key] = torch.zeros(1)
    torch.save({"state_dict": state}, tmp_path / "init.ckpt")
    runner = BallDetectionTrainingRunner()
    with pytest.raises(RuntimeError, match="state_dict"):
        runner.maybe_load_init_weights(runner.validate_runtime_config(config), module)


@pytest.mark.parametrize("checkpoint", [[], {}, {"state_dict": []}, {"state_dict": {}}])
def test_init_rejects_malformed_checkpoint(
    tmp_path: Path, checkpoint: Any,
) -> None:
    config = _config(tmp_path)
    torch.save(checkpoint, tmp_path / "init.ckpt")
    runner = BallDetectionTrainingRunner()
    with pytest.raises(ValueError, match="mapping"):
        runner.maybe_load_init_weights(
            runner.validate_runtime_config(config), BallDetectionLightningModule(config),
        )


def test_no_init_checkpoint_leaves_fresh_weights_unchanged(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.run.init_weights = None
    module = BallDetectionLightningModule(config)
    before = {key: value.clone() for key, value in module.state_dict().items()}
    runner = BallDetectionTrainingRunner()
    runner.maybe_load_init_weights(runner.validate_runtime_config(config), module)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
