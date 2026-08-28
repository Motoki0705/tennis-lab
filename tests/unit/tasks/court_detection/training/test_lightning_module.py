"""Tests for Court bundle checkpoint and prediction payload contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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
from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtConsistencyResult,
    CourtModelOutput,
    CourtModelSpec,
    CourtPoseLossKind,
    CourtPoseTrainingResult,
    CourtRawPoseOutput,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _bundle(*, kp_schema: str = "test_kp") -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema=kp_schema,
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
        loss_config=CourtLossConfig.from_mapping(
            {
                "seg": {"ce_weight": 1.0, "dice_weight": 1.0, "weight": 1.0},
                "kp": {"focal_gamma": 2.0, "weight": 1.0},
                "line": {
                    "bce_weight": 1.0,
                    "dice_weight": 1.0,
                    "pos_weight": 1.0,
                    "weight": 1.0,
                },
                "pose": {
                    "enabled": False,
                    "translation_weight": 0.0,
                    "rotation_weight": 0.0,
                    "focal_weight": 0.0,
                },
                "consistency": {
                    "enabled": False,
                    "weight": 0.0,
                    "temperature": 1.0,
                    "huber_delta": 0.01,
                    "min_depth_m": 0.1,
                    "depth_scale_m": 1.0,
                    "cheirality_weight": 0.0,
                    "warmup_fraction": 0.0,
                    "gradient_flow": "both",
                },
            }
        ),
    )


def test_bundle_snapshot_round_trip_is_order_preserving() -> None:
    bundle = _bundle()

    restored = deserialize_target_bundle(serialize_target_bundle(bundle))

    assert restored == bundle
    assert restored.kinds == ("kp", "seg", "line")


def test_scope_specific_checkpoint_bundle_mismatch_is_rejected() -> None:
    all_courts = _bundle(kp_schema="synthetic_camera_relative_kp14:gaussian_max_v1")
    target_court = _bundle(
        kp_schema="synthetic_camera_relative_kp14_target_court:gaussian_max_v1"
    )
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "data/source=synthetic_court",
                "data.source.keypoint_court_scope=target_court",
                "data/processing=kp",
            ],
        )

    assert deserialize_target_bundle(serialize_target_bundle(all_courts)) == all_courts
    assert target_court != all_courts
    with pytest.raises(ValueError, match="disagrees with its checkpoint snapshot"):
        CourtDetectionLightningModule(
            config,
            target_bundle=target_court,
            target_bundle_state=serialize_target_bundle(all_courts),
        )


@pytest.mark.parametrize(
    ("checkpoint_schema", "runtime_schema", "source"),
    [
        (
            "synthetic_symmetric_kp7:gaussian_max_v1",
            "synthetic_camera_relative_kp14:gaussian_max_v1",
            "synthetic_court_v2",
        ),
        (
            "synthetic_symmetric_kp7:gaussian_max_v1",
            "synthetic_camera_view_kp14_v3:gaussian_max_v1",
            "synthetic_court",
        ),
        (
            "synthetic_camera_relative_kp14:gaussian_max_v1",
            "synthetic_symmetric_kp7:gaussian_max_v1",
            "synthetic_court_v1",
        ),
        (
            "synthetic_camera_relative_kp14:gaussian_max_v1",
            "synthetic_camera_view_kp14_v3:gaussian_max_v1",
            "synthetic_court",
        ),
        (
            "synthetic_camera_view_kp14_v3:gaussian_max_v1",
            "synthetic_symmetric_kp7:gaussian_max_v1",
            "synthetic_court_v1",
        ),
        (
            "synthetic_camera_view_kp14_v3:gaussian_max_v1",
            "synthetic_camera_relative_kp14:gaussian_max_v1",
            "synthetic_court_v2",
        ),
    ],
)
def test_v1_v2_v3_checkpoint_bundles_are_pairwise_incompatible(
    checkpoint_schema: str,
    runtime_schema: str,
    source: str,
) -> None:
    checkpoint = _bundle(kp_schema=checkpoint_schema)
    runtime = _bundle(kp_schema=runtime_schema)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[f"data/source={source}", "data/processing=kp"],
        )

    assert checkpoint != runtime
    with pytest.raises(ValueError, match="disagrees with its checkpoint snapshot"):
        CourtDetectionLightningModule(
            config,
            target_bundle=runtime,
            target_bundle_state=serialize_target_bundle(checkpoint),
        )


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


def test_pose_loss_logs_keep_raw_weighted_and_effective_terms_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = object.__new__(CourtDetectionLightningModule)
    torch.nn.Module.__init__(module)
    logged: dict[str, float] = {}

    def record_log(name: str, value: object, **_: object) -> None:
        assert isinstance(value, torch.Tensor)
        logged[name] = float(value)

    monkeypatch.setattr(module, "log", record_log)
    raw_losses: dict[CourtPoseLossKind, torch.Tensor] = {
        "pose_translation": torch.tensor(1.0),
        "pose_rotation": torch.tensor(2.0),
        "pose_focal": torch.tensor(3.0),
    }
    weighted_losses: dict[CourtPoseLossKind, torch.Tensor] = {
        "pose_translation": torch.tensor(2.0),
        "pose_rotation": torch.tensor(6.0),
        "pose_focal": torch.tensor(12.0),
    }
    effective_weights: dict[CourtPoseLossKind, torch.Tensor] = {
        "pose_translation": torch.tensor(2.0),
        "pose_rotation": torch.tensor(3.0),
        "pose_focal": torch.tensor(4.0),
    }
    configured_weights = dict(effective_weights)
    consistency = CourtConsistencyResult(
        coordinate_loss=torch.tensor(5.0),
        cheirality_loss=torch.tensor(6.0),
        auxiliary_loss=torch.tensor(7.0),
        weighted_auxiliary_loss=torch.tensor(3.5),
        configured_weight=torch.tensor(2.0),
        effective_weight=torch.tensor(0.5),
        visible_point_count=torch.tensor(14.0),
        mean_distance_px=torch.tensor(8.0),
        invalid_depth_rate=torch.tensor(0.25),
        dense_points_xy=torch.zeros(1, 14, 2),
        pose_points_xy=torch.zeros(1, 14, 2),
        pose_depth_m=torch.ones(1, 14),
    )
    result = CourtPoseTrainingResult(
        loss=torch.tensor(20.0),
        raw_dense_loss=torch.tensor(1.0),
        direct_dense_loss=torch.tensor(0.0),
        direct_pose_loss=torch.tensor(20.0),
        raw_dense_losses={"kp": torch.tensor(1.0)},
        dense_losses={"kp": torch.tensor(0.0)},
        dense_configured_weights={"kp": torch.tensor(0.0)},
        dense_effective_weights={"kp": torch.tensor(0.0)},
        weighted_dense_losses={"kp": torch.tensor(0.0)},
        pose_losses=raw_losses,
        weighted_pose_losses=weighted_losses,
        pose_configured_weights=configured_weights,
        pose_effective_weights=effective_weights,
        consistency=consistency,
        output=CourtModelOutput(
            dense_logits={"kp": torch.zeros(1, 14, 2, 2)},
            pose=CourtRawPoseOutput(torch.zeros(1, 10)),
        ),
        decoded_pose=CourtDecodedPose(
            translation_m=torch.zeros(1, 3),
            rotation=torch.eye(3).unsqueeze(0),
            focal_px=torch.ones(1),
            log_focal=torch.zeros(1),
        ),
    )

    module._log_training_result("train", result)

    for name in raw_losses:
        assert logged[f"train/loss_{name}"] == float(raw_losses[name])
        assert logged[f"train/loss_{name}_weighted"] == float(
            weighted_losses[name]
        )
        assert logged[f"train/{name}_configured_weight"] == float(
            configured_weights[name]
        )
        assert logged[f"train/{name}_effective_weight"] == float(
            effective_weights[name]
        )
    assert logged["train/loss_direct_pose"] == 20.0
    assert logged["train/loss_direct_dense_raw"] == 1.0
    assert logged["train/loss_kp_raw"] == 1.0
    assert logged["train/loss_kp"] == 0.0
    assert logged["train/loss_kp_weighted"] == 0.0
    assert logged["train/kp_configured_weight"] == 0.0
    assert logged["train/kp_effective_weight"] == 0.0
    assert logged["train/loss_kp_pose_coordinate"] == 5.0
    assert logged["train/loss_kp_pose_cheirality"] == 6.0
    assert logged["train/loss_kp_pose_auxiliary_unweighted"] == 7.0
    assert logged["train/loss_kp_pose_auxiliary_weighted"] == 3.5
    assert logged["train/kp_pose_configured_weight"] == 2.0
    assert logged["train/kp_pose_effective_weight"] == 0.5
    assert logged["train/kp_pose_visible_point_count"] == 14.0
    assert logged["train/kp_pose_consistency_distance_px"] == 8.0
    assert logged["train/kp_pose_invalid_depth_rate"] == 0.25


def test_after_backward_uses_hierarchical_dense_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = object.__new__(CourtDetectionLightningModule)
    torch.nn.Module.__init__(module)
    model = torch.nn.Module()
    model.heads = torch.nn.ModuleDict(
        {
            "kp": torch.nn.Linear(2, 1),
            "line": torch.nn.Linear(2, 1),
        }
    )
    model.pose_head = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    module.model = model
    module.consistency_instrumented = True
    module._matrix_manifest_path = None
    logged: dict[str, float] = {}

    def record_log(name: str, value: object, **_: object) -> None:
        logged[name] = float(value)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "log", record_log)

    module.on_after_backward()

    assert logged == {
        "train/kp_gradient_finite": 1.0,
        "train/line_gradient_finite": 1.0,
        "train/pose_gradient_finite": 1.0,
    }
