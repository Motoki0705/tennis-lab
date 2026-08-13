"""Tests for Court bundle checkpoint and prediction payload contracts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import torch
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.bundle_state import (
    deserialize_target_bundle,
    serialize_target_bundle,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="test_kp",
                output_channels=2,
                channel_names=("left", "right"),
                target_dtype=torch.float32,
                precomputed=False,
            ),
            "seg": CourtTargetSpec(
                kind="seg",
                schema="test_seg",
                output_channels=3,
                channel_names=("background", "a", "b"),
                target_dtype=torch.long,
                precomputed=True,
            ),
            "line": CourtTargetSpec(
                kind="line",
                schema="test_line",
                output_channels=1,
                channel_names=("line",),
                target_dtype=torch.float32,
                precomputed=True,
            ),
        }
    )


def _adapter(bundle: CourtTargetBundleSpec) -> CourtModelIOAdapter:
    return CourtModelIOAdapter(
        CourtModelSpec(
            target_bundle=bundle,
            in_channels=3,
            short_side=32,
        ),
        loss_config=CourtLossConfig(
            seg_ce_weight=1.0,
            seg_dice_weight=1.0,
            kp_focal_gamma=2.0,
            kp_positive_weight=1.0,
            line_bce_weight=1.0,
            line_dice_weight=1.0,
            line_pos_weight=1.0,
        ),
    )


def test_bundle_snapshot_round_trip_is_order_preserving() -> None:
    bundle = _bundle()

    restored = deserialize_target_bundle(serialize_target_bundle(bundle))

    assert restored == bundle
    assert restored.kinds == ("kp", "seg", "line")


def test_runner_build_serializes_resolved_bundle_for_lightning_checkpoint() -> None:
    bundle = _bundle()
    datamodule = object.__new__(CourtDetectionDataModule)
    datamodule.target_bundle_spec = bundle
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train", overrides=["data/processing=all"])

    module = CourtDetectionTrainingRunner().build_lightning_module(
        config,
        datamodule,
    )

    assert module.target_bundle is bundle
    assert deserialize_target_bundle(module.hparams["target_bundle_state"]) == bundle
    assert module.hparams["config"] is config
    copy.deepcopy(module.hparams)
    restored = CourtDetectionLightningModule(
        module.hparams["config"],
        target_bundle_state=module.hparams["target_bundle_state"],
    )
    assert restored.target_bundle == bundle


def test_test_prediction_payload_flattens_every_selected_head() -> None:
    bundle = _bundle()
    module = object.__new__(CourtDetectionLightningModule)
    torch.nn.Module.__init__(module)
    module.__dict__["target_bundle"] = bundle
    module.model_io = _adapter(bundle)
    logits = {
        "kp": torch.zeros(2, 2, 4, 5),
        "seg": torch.zeros(2, 3, 4, 5),
        "line": torch.zeros(2, 1, 4, 5),
    }
    batch = {
        "sample_id": ["sample-a", "sample-b"],
        "image_size": torch.tensor([[4, 5], [4, 5]], dtype=torch.long),
        "targets": {
            "kp": {
                "points_xy": torch.zeros(2, 2, 4, 2),
                "point_visible": torch.ones(2, 2, 4, dtype=torch.bool),
            },
            "seg": torch.zeros(2, 4, 5, dtype=torch.long),
            "line": torch.zeros(2, 1, 4, 5),
        },
    }

    payload = module.test_prediction_payload(
        batch,
        {"logits": logits},
    )

    assert payload["sample_id"] is batch["sample_id"]
    assert payload["image_size"] is batch["image_size"]
    assert cast(torch.Tensor, payload["kp_keypoints_normalized"]).shape == (2, 2, 4, 2)
    assert cast(torch.Tensor, payload["kp_scores"]).shape == (2, 2, 4)
    assert cast(torch.Tensor, payload["kp_valid"]).shape == (2, 2, 4)
    assert cast(torch.Tensor, payload["kp_target_points_xy"]).shape == (2, 2, 4, 2)
    assert cast(torch.Tensor, payload["kp_target_point_visible"]).shape == (2, 2, 4)
    assert cast(torch.Tensor, payload["seg_mask"]).shape == (2, 4, 5)
    assert cast(torch.Tensor, payload["seg_target"]).shape == (2, 4, 5)
    assert cast(torch.Tensor, payload["line_probability"]).shape == (2, 1, 4, 5)
    assert cast(torch.Tensor, payload["line_target"]).shape == (2, 1, 4, 5)
    assert "kp_heatmaps" not in payload
    assert "seg_logits" not in payload
    assert "line_logits" not in payload
