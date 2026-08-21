"""Tests for Court bundle checkpoint and prediction payload contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.bundle_state import (
    deserialize_target_bundle,
    serialize_target_bundle,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)


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


def test_test_prediction_payload_flattens_every_selected_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    module = object.__new__(CourtDetectionLightningModule)
    torch.nn.Module.__init__(module)
    module.model_io = _adapter(bundle)
    logits = {
        "kp": torch.zeros(2, 2, 4, 5),
        "seg": torch.zeros(2, 3, 4, 5),
        "line": torch.zeros(2, 1, 4, 5),
    }
    batch = {
        "image_size": torch.tensor([[4, 5], [4, 5]], dtype=torch.long),
    }

    payload = module.test_prediction_payload(
        batch,
        {"logits": logits},
    )

    expected_shapes = {
        "image_size": (2, 2),
        "kp_keypoints_normalized": (2, 2, 4, 2),
        "kp_scores": (2, 2, 4),
        "kp_valid": (2, 2, 4),
        "kp_heatmaps": (2, 2, 4, 5),
        "seg_mask": (2, 4, 5),
        "seg_logits": (2, 3, 4, 5),
        "line_probability": (2, 1, 4, 5),
        "line_logits": (2, 1, 4, 5),
    }
    assert set(payload) == set(expected_shapes)
    assert all(isinstance(value, np.ndarray) for value in payload.values())
    assert {key: value.shape for key, value in payload.items()} == expected_shapes
    np.testing.assert_array_equal(payload["image_size"], [[4, 5], [4, 5]])
    np.testing.assert_array_equal(payload["kp_heatmaps"], np.zeros((2, 2, 4, 5)))
    np.testing.assert_array_equal(payload["seg_logits"], np.zeros((2, 3, 4, 5)))
    np.testing.assert_array_equal(payload["line_logits"], np.zeros((2, 1, 4, 5)))

    module._reset_test_prediction_buffer()
    monkeypatch.setattr(module, "_test_predictions_dir", lambda: tmp_path)
    module.collect_test_predictions(batch, {"logits": logits})
    saved = module.save_test_predictions()

    assert saved == tmp_path / "pred_test.npz"
    with np.load(saved, allow_pickle=False) as archive:
        assert set(archive.files) == {*expected_shapes, "scene_ids"}
        assert {key: archive[key].shape for key in expected_shapes} == expected_shapes
        np.testing.assert_array_equal(
            archive["scene_ids"],
            ["sample_000000", "sample_000001"],
        )
