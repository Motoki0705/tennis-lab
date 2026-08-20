"""Integration contract for the baseline-matched Issue #753 training data."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from hydra import compose, initialize_config_dir

from experiments.blcs.issue_753.baseline_matched import BaselineMatchedDataModule
from src.tasks.blcs.model_io.factory import (
    TrackQueryBoundModelIO,
    compose_blcs_model_io,
)
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)

_CONFIG_DIR = Path("experiments/blcs/issue_753/configs").resolve()


def test_baseline_matched_batch_runs_current_fixed_query_model() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "data.split_sizes.train=2",
                "data.split_sizes.val=2",
                "data.batch_size=2",
            ],
        )
    datamodule = BaselineMatchedDataModule(config)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    binding = compose_blcs_model_io(config)
    lightning_module = BLCSTrackingLightningModule(
        config,
        model_io=cast("TrackQueryBoundModelIO", binding),
    )
    prepared = binding.adapter.build_training_batch(batch)
    prediction = binding.decode_output(binding.execute_call(prepared.call))

    assert "model_io" not in lightning_module.hparams
    assert batch["ball_uv"].shape == (2, 3, 12, 4, 2)
    assert batch["target_position"].shape == (2, 12, 4, 3)
    assert prediction.position.shape == (2, 12, 4, 3)
    assert prediction.presence_logits.shape == (2, 12, 4)
    assert torch.isfinite(prediction.position).all()
    assert torch.isfinite(prediction.presence_logits).all()
