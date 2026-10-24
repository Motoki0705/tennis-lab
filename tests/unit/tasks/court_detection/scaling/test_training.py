"""Retention configuration for fixed-update Court checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.court_detection.scaling.training import step_checkpoints
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
)

pytestmark = pytest.mark.unit
_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


@pytest.fixture
def config(tmp_path: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_data_scaling",
            overrides=[
                f"paths.project_root={tmp_path}",
                "data.source.manifest=court_scale/fixture.dataset.json",
                "mixed.sources.tennis_court_detector.registry=real/registry.json",
                "training.warmup_steps=1",
                "training.step_schedule.max_steps=8",
                "training.step_schedule.validation_every_n_steps=2",
                "training.step_schedule.checkpoint_every_n_steps=4",
                "run.output_dir=court_detection/train/retention-fixture/s42",
            ],
        )


@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_step_checkpoints_obey_retention(
    config: DictConfig, tmp_path: Path, save_top_k: int
) -> None:
    config.training.checkpoint.save_top_k = save_top_k
    logger = TensorBoardLogger(save_dir=str(tmp_path), name="logs", version=0)

    periodic, best = step_checkpoints(config, logger)

    assert periodic.dirpath == best.dirpath == str(Path(logger.log_dir) / "checkpoints")
    assert periodic.filename == "update-{step:08d}"
    assert periodic.save_top_k == -1 and periodic.monitor is None
    assert best.filename == "best-val-{step:08d}"
    assert best.save_top_k == save_top_k
    assert best.monitor == "val/loss" and best.mode == "min"
    assert best.every_n_epochs == 1
    assert not periodic.auto_insert_metric_name and not best.auto_insert_metric_name
    assert not periodic.save_last and not best.save_last


@pytest.mark.parametrize("value", [-2, -100])
def test_step_checkpoints_reject_unsupported_retention(
    config: DictConfig, tmp_path: Path, value: int
) -> None:
    config.training.checkpoint.save_top_k = value
    logger = TensorBoardLogger(save_dir=str(tmp_path), name="logs", version=0)
    with pytest.raises(SemanticConfigurationError, match="checkpoint.save_top_k"):
        step_checkpoints(config, logger)


@pytest.mark.parametrize("value", [None, True, False, -1.0, 1.5, "1", "all"])
def test_step_checkpoints_reject_retention_type_coercion(
    config: DictConfig, tmp_path: Path, value: object
) -> None:
    config.training.checkpoint.save_top_k = value
    logger = TensorBoardLogger(save_dir=str(tmp_path), name="logs", version=0)
    with pytest.raises(ConfigurationTypeError, match="checkpoint.save_top_k"):
        step_checkpoints(config, logger)


def test_step_checkpoints_require_retention(config: DictConfig, tmp_path: Path) -> None:
    with open_dict(config.training.checkpoint):
        del config.training.checkpoint.save_top_k
    logger = TensorBoardLogger(save_dir=str(tmp_path), name="logs", version=0)
    with pytest.raises(MissingConfigurationKeyError, match="checkpoint.save_top_k"):
        step_checkpoints(config, logger)


def test_scaling_preset_retains_all_checkpoints(
    config: DictConfig, tmp_path: Path
) -> None:
    logger = TensorBoardLogger(save_dir=str(tmp_path), name="logs", version=0)
    assert config.training.checkpoint.save_top_k == -1
    assert [callback.save_top_k for callback in step_checkpoints(config, logger)] == [
        -1,
        -1,
    ]
