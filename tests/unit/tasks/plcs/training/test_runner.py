"""Unit tests for PLCS-specific weight initialization contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.plcs.model_io import write_plcs_checkpoint_court_keypoints
from src.tasks.plcs.training.runner import PLCSTrainingRunner
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import add_court_coordinate_normalization


def _module(
    tmp_path: Path,
    *,
    fine_tune_mode: Literal["all", "presence_head"],
) -> PLCSTrackingLightningModule:
    config_name = (
        "train_tracking_pose_presence_head"
        if fine_tune_mode == "presence_head"
        else "train_tracking_pose"
    )
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name=config_name,
            overrides=[
                "model.hidden_dim=32",
                "model.num_heads=4",
                "model.ffn_dim=64",
                "model.rope_dim=4",
                "model.mhc.coefficient_dim=16",
                "training.compile.enabled=false",
            ],
        )
    config.paths.project_root = str(tmp_path)
    config.paths.checkpoint_root = "checkpoints"
    config.run.init_weights = "source.ckpt"
    return PLCSTrackingLightningModule(config)


def _write_checkpoint(
    module: PLCSTrackingLightningModule,
    *,
    omitted_prefix: str | None = None,
) -> Path:
    state_dict = {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
        if omitted_prefix is None or not key.startswith(omitted_prefix)
    }
    checkpoint: dict[str, object] = {"state_dict": state_dict}
    add_court_coordinate_normalization(
        checkpoint,
        artifact="PLCS test checkpoint",
    )
    write_plcs_checkpoint_court_keypoints(
        checkpoint,
        module.plcs_runtime.court_keypoint_contract,
    )
    checkpoint_path = cast(
        "Path | None",
        module.plcs_runtime.shared.run.init_weights,
    )
    assert checkpoint_path is not None
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.mark.parametrize(
    "omitted_prefix",
    ["model.presence_head.", "model.canonical_pose_head."],
)
def test_presence_head_init_rejects_any_missing_model_state(
    tmp_path: Path,
    omitted_prefix: str,
) -> None:
    module = _module(tmp_path, fine_tune_mode="presence_head")
    checkpoint_path = _write_checkpoint(module, omitted_prefix=omitted_prefix)

    with pytest.raises(
        RuntimeError,
        match=r"must initialize every model parameter and buffer.*missing",
    ):
        PLCSTrainingRunner().maybe_load_init_weights(
            module.plcs_runtime.shared,
            module,
        )

    assert checkpoint_path.exists()


def test_presence_head_init_accepts_complete_model_state(tmp_path: Path) -> None:
    module = _module(tmp_path, fine_tune_mode="presence_head")
    _write_checkpoint(module)

    PLCSTrainingRunner().maybe_load_init_weights(
        module.plcs_runtime.shared,
        module,
    )


def test_all_mode_keeps_legacy_partial_init_behavior(tmp_path: Path) -> None:
    module = _module(tmp_path, fine_tune_mode="all")
    _write_checkpoint(module, omitted_prefix="model.canonical_pose_head.")

    PLCSTrainingRunner().maybe_load_init_weights(
        module.plcs_runtime.shared,
        module,
    )
