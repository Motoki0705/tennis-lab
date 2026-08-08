"""Base-specific fixtures for shared training configuration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def make_training_config(tmp_path: Path):
    """Build a complete shared training config with explicit test values."""

    def _factory(
        *,
        run: dict[str, Any] | None = None,
        trainer: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_config: dict[str, Any] = {
            "output_dir": "run",
            "seed": 1,
            "gpus": 0,
            "resume": None,
            "init_weights": None,
            "fast_dev_run": False,
            "dry_run": False,
            "test_after_fit": False,
        }
        run_config.update(run or {})
        trainer_config: dict[str, Any] = {
            "max_epochs": 2,
            "gradient_clip_val": None,
            "deterministic": True,
            "precision": "32-true",
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 1,
            "accumulate_grad_batches": 1,
            "reload_dataloaders_every_n_epochs": 0,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "benchmark": False,
        }
        trainer_config.update(trainer or {})
        training_config: dict[str, Any] = {
            "trainer": trainer_config,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "warmup_epochs": None,
            "min_lr": 0.0,
            "steps_per_epoch": 1,
            "optimizer": {"betas": [0.9, 0.999]},
            "checkpoint": {
                "enabled": False,
                "filename": "model-{epoch}",
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": False,
            },
            "early_stopping": {
                "enabled": False,
                "monitor": "val/loss",
                "mode": "min",
                "patience": 1,
                "min_delta": 0.0,
                "check_on_train_epoch_end": False,
            },
            "lr_monitor": {"enabled": False, "interval": "step"},
            "qualitative_logging": {
                "enabled": False,
                "every_n_epochs": 1,
                "num_samples": 1,
                "selection_mode": "random",
                "selected_indices": None,
            },
            "gan": {
                "enabled": False,
                "target_weight": 0.0,
                "warmup_epochs": 1,
                "generator_gradient_clip_val": None,
                "discriminator_gradient_clip_val": None,
                "transition": {"start_epoch": 0},
            },
            "matmul_precision": "high",
            "allow_tf32": False,
        }
        training_config.update(training or {})
        return {
            "paths": {
                "project_root": str(tmp_path),
                "data_root": "data",
                "checkpoint_root": str(tmp_path),
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "external",
            },
            "run": run_config,
            "training": training_config,
        }

    return _factory
