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
    fine_tune_mode: Literal["all", "presence_head", "presence_competition"],
    enable_competition: bool | None = None,
) -> PLCSTrackingLightningModule:
    config_name = {
        "all": "train_tracking_pose",
        "presence_head": "train_tracking_pose_presence_head",
        "presence_competition": "train_tracking_pose_presence_competition",
    }[fine_tune_mode]
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    overrides = [
        "model.hidden_dim=32",
        "model.num_heads=4",
        "model.ffn_dim=64",
        "model.rope_dim=4",
        "model.mhc.coefficient_dim=16",
        "training.compile.enabled=false",
    ]
    if enable_competition is True:
        overrides.append("model.presence_competition=deepsets")
    elif enable_competition is False:
        overrides.append("model.presence_competition=none")
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name=config_name,
            overrides=overrides,
        )
    config.paths.project_root = str(tmp_path)
    config.paths.checkpoint_root = "checkpoints"
    config.run.init_weights = "source.ckpt"
    return PLCSTrackingLightningModule(config)


def _write_checkpoint(
    module: PLCSTrackingLightningModule,
    *,
    omitted_prefix: str | None = None,
    omitted_key: str | None = None,
    unexpected_key: str | None = None,
) -> Path:
    state_dict = {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
        if (omitted_prefix is None or not key.startswith(omitted_prefix))
        and key != omitted_key
    }
    if unexpected_key is not None:
        state_dict[unexpected_key] = torch.tensor(1.0)
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


def test_presence_competition_accepts_only_complete_legacy_branch_omission(
    tmp_path: Path,
) -> None:
    module = _module(tmp_path, fine_tune_mode="presence_competition")
    branch = module.model.presence_competition
    with torch.no_grad():
        branch.output_projection.weight.fill_(2.0)
        branch.output_projection.bias.fill_(3.0)
    _write_checkpoint(
        module,
        omitted_prefix="model.presence_competition.",
    )

    PLCSTrainingRunner().maybe_load_init_weights(
        module.plcs_runtime.shared,
        module,
    )

    assert torch.count_nonzero(branch.output_projection.weight) == 0
    assert torch.count_nonzero(branch.output_projection.bias) == 0


@pytest.mark.parametrize(
    "omitted_key",
    [
        "model.presence_competition.output_projection.weight",
        "model.presence_head.weight",
        "model.canonical_pose_head.mlp.8.weight",
    ],
)
def test_presence_competition_rejects_partial_or_nonbranch_missing_state(
    tmp_path: Path,
    omitted_key: str,
) -> None:
    module = _module(tmp_path, fine_tune_mode="presence_competition")
    _write_checkpoint(module, omitted_key=omitted_key)

    with pytest.raises(RuntimeError, match="disallowed/partial missing keys"):
        PLCSTrainingRunner().maybe_load_init_weights(
            module.plcs_runtime.shared,
            module,
        )


def test_presence_competition_rejects_unexpected_state(tmp_path: Path) -> None:
    module = _module(tmp_path, fine_tune_mode="presence_competition")
    _write_checkpoint(module, unexpected_key="model.legacy_presence_extra")

    with pytest.raises(RuntimeError, match="unexpected state keys"):
        PLCSTrainingRunner().maybe_load_init_weights(
            module.plcs_runtime.shared,
            module,
        )


@pytest.mark.parametrize(
    ("omitted_prefix", "unexpected_key"),
    [
        ("model.presence_competition.", None),
        (None, "model.legacy_presence_extra"),
    ],
)
def test_enabled_branch_outside_explicit_migration_mode_is_strict(
    tmp_path: Path,
    omitted_prefix: str | None,
    unexpected_key: str | None,
) -> None:
    module = _module(
        tmp_path,
        fine_tune_mode="all",
        enable_competition=True,
    )
    _write_checkpoint(
        module,
        omitted_prefix=omitted_prefix,
        unexpected_key=unexpected_key,
    )

    with pytest.raises(RuntimeError, match="must match exactly"):
        PLCSTrainingRunner().maybe_load_init_weights(
            module.plcs_runtime.shared,
            module,
        )


def test_presence_competition_accepts_strict_enabled_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    module = _module(tmp_path, fine_tune_mode="presence_competition")
    branch = module.model.presence_competition
    with torch.no_grad():
        branch.output_projection.weight.fill_(0.75)
        branch.output_projection.bias.fill_(-0.25)
    expected = {
        key: value.detach().clone()
        for key, value in branch.state_dict().items()
    }
    _write_checkpoint(module)
    with torch.no_grad():
        branch.output_projection.weight.zero_()
        branch.output_projection.bias.zero_()

    PLCSTrainingRunner().maybe_load_init_weights(
        module.plcs_runtime.shared,
        module,
    )

    for key, value in branch.state_dict().items():
        assert torch.equal(value, expected[key])
