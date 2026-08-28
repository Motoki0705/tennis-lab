"""Unit contracts for the unified hierarchical Court model-I/O seam."""

from __future__ import annotations

import torch

from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)
from src.tasks.court_detection.model_io.adapters import (
    CourtModelIOAdapter,
    CourtPoseModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtModelOutput,
    CourtModelSpec,
)
from src.tasks.court_detection.models.pose_head import CourtRawPoseOutput


def _loss(
    *,
    pose: bool = False,
    dense_weight: float = 1.0,
    pose_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    consistency_weight: float = 0.0,
    consistency_warmup_fraction: float = 0.0,
) -> CourtLossConfig:
    translation_weight, rotation_weight, focal_weight = pose_weights
    return CourtLossConfig.from_mapping(
        {
            "seg": {
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "weight": dense_weight,
            },
            "kp": {"focal_gamma": 2.0, "weight": dense_weight},
            "line": {
                "bce_weight": 1.0,
                "dice_weight": 1.0,
                "pos_weight": 1.0,
                "weight": dense_weight,
            },
            "pose": {
                "enabled": pose,
                "translation_weight": translation_weight if pose else 0.0,
                "rotation_weight": rotation_weight if pose else 0.0,
                "focal_weight": focal_weight if pose else 0.0,
            },
            "consistency": {
                "enabled": consistency_weight > 0.0,
                "weight": consistency_weight,
                "temperature": 1.0,
                "huber_delta": 0.01,
                "min_depth_m": 0.1,
                "depth_scale_m": 1.0,
                "cheirality_weight": 0.0,
                "warmup_fraction": consistency_warmup_fraction,
                "gradient_flow": "both",
            },
        }
    )


def _bundle(kind: CourtTargetKind) -> CourtTargetBundleSpec:
    if kind == "line":
        spec = CourtTargetSpec(
            kind="line",
            schema="line",
            output_channels=1,
            channel_names=("line",),
            target_dtype=torch.float32,
            precomputed=False,
        )
    else:
        spec = CourtTargetSpec(
            kind="kp",
            schema="kp",
            output_channels=14,
            channel_names=tuple(str(i) for i in range(14)),
            target_dtype=torch.float32,
            precomputed=False,
        )
    return CourtTargetBundleSpec({kind: spec})


def test_dense_output_contract_keeps_legacy_result_without_pose() -> None:
    bundle = _bundle("line")
    adapter = CourtModelIOAdapter(
        CourtModelSpec(bundle, in_channels=3, short_side=16),
        loss_config=_loss(),
    )
    batch = {
        "image": torch.zeros(1, 3, 4, 5),
        "targets": {"line": torch.zeros(1, 1, 4, 5)},
    }
    call = adapter.prepare_training_batch(batch)
    result = adapter.training_result(
        CourtModelOutput({"line": torch.zeros(1, 1, 4, 5)}),
        call,
    )
    assert result.__class__.__name__ == "CourtTrainingResult"
    assert result.loss.shape == ()
    assert not hasattr(result, "pose_configured_weights")


def test_pose_only_objective_keeps_kp_contract_without_dense_gradient() -> None:
    bundle = _bundle("kp")
    adapter = CourtPoseModelIOAdapter(
        CourtModelSpec(bundle, in_channels=3, short_side=16),
        loss_config=_loss(
            pose=True,
            dense_weight=0.0,
            pose_weights=(2.0, 3.0, 4.0),
        ),
    )
    target_raw_pose = torch.tensor(
        [[0.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0]]
    )
    predicted_raw_pose = torch.tensor(
        [[1.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.2]],
        requires_grad=True,
    )
    kp_logits = torch.zeros(1, 14, 4, 5, requires_grad=True)
    batch = {
        "image": torch.zeros(1, 3, 4, 5),
        "targets": {
            "kp": {
                "heatmap": torch.zeros(1, 14, 4, 5),
                "points_xy": torch.zeros(1, 14, 1, 2),
                "point_visible": torch.ones(1, 14, 1, dtype=torch.bool),
                "physical_indices": torch.arange(14).view(1, 14, 1),
            }
        },
        "pose_target": {
            "translation_m": torch.tensor([[0.0, -20.0, 10.0]]),
            "rotation": torch.eye(3).unsqueeze(0),
            "log_focal": torch.tensor([4.0]),
            "intrinsics": torch.tensor(
                [[[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]]
            ),
            "semantic_to_physical": torch.arange(14).view(1, 14),
            "raw_pose10d": target_raw_pose,
        },
        "image_size": torch.tensor([[4, 5]]),
    }
    call = adapter.prepare_training_batch(batch)
    result = adapter.training_result(
        CourtModelOutput(
            {"kp": kp_logits},
            CourtRawPoseOutput(predicted_raw_pose),
        ),
        call,
    )

    assert result.direct_dense_loss.item() == 0.0
    assert result.raw_dense_loss.item() > 0.0
    assert result.raw_dense_losses["kp"].item() > 0.0
    assert result.dense_losses["kp"].item() == 0.0
    assert result.weighted_dense_losses["kp"].item() == 0.0
    assert result.dense_configured_weights["kp"].item() == 0.0
    assert result.dense_effective_weights["kp"].item() == 0.0
    assert result.direct_pose_loss.item() > 0.0
    assert set(result.pose_losses) == {
        "pose_translation",
        "pose_rotation",
        "pose_focal",
    }
    assert set(result.weighted_pose_losses) == set(result.pose_losses)
    assert set(result.pose_configured_weights) == set(result.pose_losses)
    assert set(result.pose_effective_weights) == set(result.pose_losses)
    expected_weights = {
        "pose_translation": 2.0,
        "pose_rotation": 3.0,
        "pose_focal": 4.0,
    }
    for name, raw_loss in result.pose_losses.items():
        torch.testing.assert_close(
            result.pose_configured_weights[name],
            raw_loss.new_tensor(expected_weights[name]),
        )
        torch.testing.assert_close(
            result.pose_effective_weights[name],
            raw_loss.new_tensor(expected_weights[name]),
        )
        torch.testing.assert_close(
            result.weighted_pose_losses[name],
            raw_loss * result.pose_effective_weights[name],
        )
    torch.testing.assert_close(
        result.direct_pose_loss,
        torch.stack(tuple(result.weighted_pose_losses.values())).sum(),
    )
    torch.testing.assert_close(result.loss, result.direct_pose_loss)
    assert result.consistency is None

    result.loss.backward()
    assert kp_logits.grad is not None
    assert torch.count_nonzero(kp_logits.grad).item() == 0
    assert predicted_raw_pose.grad is not None
    assert bool(torch.isfinite(predicted_raw_pose.grad).all())
    assert torch.count_nonzero(predicted_raw_pose.grad).item() > 0


def test_consistency_result_keeps_configured_and_warmup_effective_weights() -> None:
    bundle = _bundle("kp")
    adapter = CourtPoseModelIOAdapter(
        CourtModelSpec(bundle, in_channels=3, short_side=16),
        loss_config=_loss(
            pose=True,
            consistency_weight=2.0,
            consistency_warmup_fraction=0.5,
        ),
    )
    raw_pose = torch.tensor(
        [[0.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0]]
    )
    batch = {
        "image": torch.zeros(1, 3, 4, 5),
        "targets": {
            "kp": {
                "heatmap": torch.zeros(1, 14, 4, 5),
                "points_xy": torch.zeros(1, 14, 1, 2),
                "point_visible": torch.ones(1, 14, 1, dtype=torch.bool),
                "physical_indices": torch.arange(14).view(1, 14, 1),
            }
        },
        "pose_target": {
            "translation_m": torch.tensor([[0.0, -20.0, 10.0]]),
            "rotation": torch.eye(3).unsqueeze(0),
            "log_focal": torch.tensor([4.0]),
            "intrinsics": torch.tensor(
                [[[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]]
            ),
            "semantic_to_physical": torch.arange(14).view(1, 14),
            "raw_pose10d": raw_pose,
        },
        "image_size": torch.tensor([[4, 5]]),
        "content_size_hw": torch.tensor([[4, 5]]),
    }
    call = adapter.prepare_training_batch(batch)
    result = adapter.training_result(
        CourtModelOutput(
            {"kp": torch.zeros(1, 14, 4, 5)},
            CourtRawPoseOutput(raw_pose.clone()),
        ),
        call,
        progress_fraction=0.75,
    )

    consistency = result.consistency
    assert consistency is not None
    assert consistency.configured_weight.item() == 2.0
    assert consistency.effective_weight.item() == 1.0
    assert (
        consistency.configured_weight.item()
        != consistency.effective_weight.item()
    )
    torch.testing.assert_close(
        consistency.weighted_auxiliary_loss,
        consistency.auxiliary_loss * consistency.effective_weight,
    )
    torch.testing.assert_close(
        result.loss,
        result.direct_dense_loss
        + result.direct_pose_loss
        + consistency.weighted_auxiliary_loss,
    )
