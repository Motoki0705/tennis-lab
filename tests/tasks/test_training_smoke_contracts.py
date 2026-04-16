from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning_fabric.utilities.exceptions import MisconfigurationException
from omegaconf import DictConfig

from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.tasks.event_detection.training.runner import EventDetectionTrainingRunner
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.tasks.trajectory_completion.training.runner import TrajectoryCompletionTrainingRunner


@dataclass(frozen=True)
class SmokeCase:
    name: str
    runner: type[Any]
    config_dir: Path
    config_name: str
    overrides: tuple[str, ...]
    supports_test_phase: bool
    expect_qualitative: bool


REPO_ROOT = Path(__file__).resolve().parents[2]


SMOKE_CASES = (
    SmokeCase(
        name="blcs",
        runner=BLCSTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/blcs/configs",
        config_name="train",
        overrides=(
            "run.gpus=0",
            "data.batch_size=1",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.scene_sampler=false",
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
            "training.trainer.max_epochs=1",
            "training.trainer.limit_train_batches=1",
            "training.trainer.limit_val_batches=1",
            "training.trainer.num_sanity_val_steps=0",
            "training.trainer.precision=32-true",
            "training.trainer.enable_progress_bar=false",
            "training.trainer.enable_model_summary=false",
            "training.trainer.log_every_n_steps=1",
            "training.checkpoint.save_top_k=1",
            "training.checkpoint.save_last=true",
            "training.qualitative_logging.enabled=true",
            "training.qualitative_logging.every_n_epochs=1",
            "training.qualitative_logging.num_samples=1",
            "training.qualitative_logging.selection_mode=fixed_indices",
            "training.qualitative_logging.selected_indices=[0]",
        ),
        supports_test_phase=True,
        expect_qualitative=False,
    ),
    SmokeCase(
        name="plcs",
        runner=PLCSTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/plcs/configs",
        config_name="train",
        overrides=(
            "run.gpus=0",
            "data.batch_size=1",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.scene_sampler=false",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "training.trainer.max_epochs=1",
            "training.trainer.limit_train_batches=1",
            "training.trainer.limit_val_batches=1",
            "training.trainer.num_sanity_val_steps=0",
            "training.trainer.precision=32-true",
            "training.trainer.enable_progress_bar=false",
            "training.trainer.enable_model_summary=false",
            "training.trainer.log_every_n_steps=1",
            "training.checkpoint.save_top_k=1",
            "training.checkpoint.save_last=true",
            "training.qualitative_logging.enabled=true",
            "training.qualitative_logging.every_n_epochs=1",
            "training.qualitative_logging.num_samples=1",
            "training.qualitative_logging.selection_mode=fixed_indices",
            "training.qualitative_logging.selected_indices=[0]",
        ),
        supports_test_phase=True,
        expect_qualitative=True,
    ),
    SmokeCase(
        name="event_detection_uv_fit_only_no_test_dataloader",
        runner=EventDetectionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/event_detection/configs",
        config_name="train_uv",
        overrides=(
            "run.gpus=0",
            "data.batch_size=1",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.scene_sampler=false",
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
            "training.trainer.max_epochs=1",
            "training.trainer.limit_train_batches=1",
            "training.trainer.limit_val_batches=1",
            "training.trainer.num_sanity_val_steps=0",
            "training.trainer.precision=32-true",
            "training.trainer.enable_progress_bar=false",
            "training.trainer.enable_model_summary=false",
            "training.trainer.log_every_n_steps=1",
            "training.checkpoint.save_top_k=1",
            "training.checkpoint.save_last=true",
            "training.qualitative_logging.enabled=true",
            "training.qualitative_logging.every_n_epochs=1",
            "training.qualitative_logging.num_samples=1",
            "training.qualitative_logging.selection_mode=fixed_indices",
            "training.qualitative_logging.selected_indices=[0]",
        ),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
    SmokeCase(
        name="event_detection_3d_fit_only_no_test_dataloader",
        runner=EventDetectionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/event_detection/configs",
        config_name="train_3d",
        overrides=(
            "run.gpus=0",
            "data.batch_size=1",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.scene_sampler=false",
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
            "training.trainer.max_epochs=1",
            "training.trainer.limit_train_batches=1",
            "training.trainer.limit_val_batches=1",
            "training.trainer.num_sanity_val_steps=0",
            "training.trainer.precision=32-true",
            "training.trainer.enable_progress_bar=false",
            "training.trainer.enable_model_summary=false",
            "training.trainer.log_every_n_steps=1",
            "training.checkpoint.save_top_k=1",
            "training.checkpoint.save_last=true",
            "training.qualitative_logging.enabled=true",
            "training.qualitative_logging.every_n_epochs=1",
            "training.qualitative_logging.num_samples=1",
            "training.qualitative_logging.selection_mode=fixed_indices",
            "training.qualitative_logging.selected_indices=[0]",
        ),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
    SmokeCase(
        name="trajectory_completion_fit_only_no_test_dataloader",
        runner=TrajectoryCompletionTrainingRunner,
        config_dir=REPO_ROOT / "src/tasks/trajectory_completion/configs",
        config_name="train",
        overrides=(
            "run.gpus=0",
            "data.batch_size=1",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.scene_sampler=false",
            "data.seq_len_range=[16,16]",
            "model.hidden_dim=32",
            "model.num_ball_layers=2",
            "model.num_query_layers=1",
            "model.num_heads=4",
            "model.ffn_dim=64",
            "model.max_seq_len=64",
            "training.trainer.max_epochs=1",
            "training.trainer.limit_train_batches=1",
            "training.trainer.limit_val_batches=1",
            "training.trainer.num_sanity_val_steps=0",
            "training.trainer.precision=32-true",
            "training.trainer.enable_progress_bar=false",
            "training.trainer.enable_model_summary=false",
            "training.trainer.log_every_n_steps=1",
            "training.checkpoint.save_top_k=1",
            "training.checkpoint.save_last=true",
            "training.qualitative_logging.enabled=true",
            "training.qualitative_logging.every_n_epochs=1",
            "training.qualitative_logging.num_samples=1",
            "training.qualitative_logging.selection_mode=fixed_indices",
            "training.qualitative_logging.selected_indices=[0]",
        ),
        supports_test_phase=False,
        expect_qualitative=True,
    ),
)


def _compose_config(case: SmokeCase, output_dir: Path) -> DictConfig:
    GlobalHydra.instance().clear()
    overrides = [_normalize_override(f"run.output_dir={output_dir.as_posix()}")]
    overrides.extend(_normalize_override(override) for override in case.overrides)
    with initialize_config_dir(version_base="1.3", config_dir=str(case.config_dir)):
        return compose(config_name=case.config_name, overrides=overrides)


def _normalize_override(override: str) -> str:
    if override.startswith(("+", "~")):
        return override
    key, separator, value = override.partition("=")
    if not separator:
        return override
    return f"++{key}={value}"


def _assert_common_artifacts(output_dir: Path, *, expect_qualitative: bool) -> None:
    assert (output_dir / "config.yaml").is_file()

    log_dir = output_dir / "logs" / "version_0"
    assert log_dir.is_dir()

    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert event_files

    checkpoint_dir = log_dir / "checkpoints"
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "last.ckpt").is_file()

    saved_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    assert saved_checkpoints

    qualitative_dir = log_dir / "qualitative" / "epoch_0000"
    if expect_qualitative:
        qualitative_images = list(qualitative_dir.glob("*.png"))
        assert qualitative_images
    else:
        assert not qualitative_dir.exists()


@pytest.mark.integration
@pytest.mark.local_data
@pytest.mark.slow
@pytest.mark.parametrize("case", SMOKE_CASES, ids=[case.name for case in SMOKE_CASES])
def test_scene_training_smoke_contracts(case: SmokeCase, tmp_path: Path) -> None:
    output_dir = tmp_path / case.name
    config = _compose_config(case, output_dir)

    runner = case.runner()
    runner.seed_everything(config)
    runner.apply_runtime_settings(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    runner.save_config(config, output_dir)

    datamodule = runner.build_datamodule(config)
    lightning_module = runner.build_lightning_module(config, datamodule)
    logger = runner.build_logger(config, output_dir)
    callbacks = runner.build_callbacks(config, datamodule, logger)
    trainer = runner.build_trainer(config, callbacks, logger)

    trainer.fit(lightning_module, datamodule=datamodule)

    if case.supports_test_phase:
        trainer.test(lightning_module, datamodule=datamodule)
    else:
        with pytest.raises(
            (AttributeError, MisconfigurationException, NotImplementedError, RuntimeError)
        ):
            trainer.test(lightning_module, datamodule=datamodule)

    _assert_common_artifacts(output_dir, expect_qualitative=case.expect_qualitative)
