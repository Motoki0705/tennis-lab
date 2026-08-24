"""Pre-side-effect checkpoint validation for Court query training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.court_detection.configuration import CourtQueryLossConfig
from src.tasks.court_detection.data.bundle_state import (
    serialize_query_checkpoint_state,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _config() -> object:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "data/source=synthetic_court",
                "data.source.keypoint_court_scope=target_court",
                "data/processing=all",
                "data/augmentation=pose_safe",
                "loss=query_joint_both",
                "model=query_encoder",
                "model.heads.dense_targets=[kp,seg,line]",
            ],
        )
    with open_dict(config.run):
        config.run.resume = "query.ckpt"
    return config


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema=(
                    "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"
                ),
                output_channels=14,
                channel_names=tuple(f"kp_{index}" for index in range(14)),
                target_dtype=torch.float32,
                precomputed=False,
            ),
            "seg": CourtTargetSpec(
                kind="seg",
                schema="court_cell_segmentation_v1",
                output_channels=7,
                channel_names=tuple(f"cell_{index}" for index in range(7)),
                target_dtype=torch.long,
                precomputed=True,
            ),
            "line": CourtTargetSpec(
                kind="line",
                schema="court_line_binary_v1",
                output_channels=1,
                channel_names=("line",),
                target_dtype=torch.float32,
                precomputed=True,
            ),
        }
    )


def _checkpoint(config: object, *, mismatched: bool) -> dict[str, object]:
    from src.tasks.court_detection.configuration import CourtTrainingConfig

    runtime = CourtTrainingConfig.from_config(config)
    assert isinstance(runtime.loss, CourtQueryLossConfig)
    consistency = runtime.loss.consistency
    if mismatched:
        consistency = replace(consistency, temperature=0.5)
    return {
        "hyper_parameters": {
            "query_checkpoint_state": serialize_query_checkpoint_state(
                _bundle(),
                loss_config_name=runtime.loss.name,
                pose_supervision=True,
                consistency=consistency,
            )
        }
    }


def test_runner_accepts_exact_enabled_checkpoint_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    checkpoint = _checkpoint(config, mismatched=False)
    monkeypatch.setattr(
        "src.tasks.court_detection.training.runner._load_checkpoint",
        lambda _: checkpoint,
    )

    CourtDetectionTrainingRunner().prepare_config(config)


def test_runner_rejects_enabled_checkpoint_mismatch_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    checkpoint = _checkpoint(config, mismatched=True)
    monkeypatch.setattr(
        "src.tasks.court_detection.training.runner._load_checkpoint",
        lambda _: checkpoint,
    )

    with pytest.raises(ValueError, match="supervision identity"):
        CourtDetectionTrainingRunner().prepare_config(config)
