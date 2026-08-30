"""Unit coverage for synthetic-only pose supervision in mixed batches."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtModelIOError,
    CourtModelOutput,
    CourtModelSpec,
)
from src.tasks.court_detection.model_io.mixed_adapter import (
    MixedCourtPoseModelIOAdapter,
)
from src.tasks.court_detection.models.pose_head import CourtRawPoseOutput
from src.utils.schema.court import GROUND_COURT_KP_NAMES

pytestmark = pytest.mark.unit


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1",
                output_channels=14,
                channel_names=GROUND_COURT_KP_NAMES,
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


def _loss(*, consistency_enabled: bool) -> CourtLossConfig:
    return CourtLossConfig.from_mapping(
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
                "enabled": True,
                "translation_weight": 1.0,
                "rotation_weight": 1.0,
                "focal_weight": 1.0,
            },
            "consistency": {
                "enabled": consistency_enabled,
                "weight": 1.0 if consistency_enabled else 0.0,
                "temperature": 1.0,
                "huber_delta": 0.01,
                "min_depth_m": 0.1,
                "depth_scale_m": 1.0,
                "cheirality_weight": 0.0,
                "warmup_fraction": 0.0,
                "gradient_flow": "both",
            },
        }
    )


def _pose_target() -> dict[str, torch.Tensor]:
    raw_pose = torch.tensor([0.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0])
    return {
        "translation_m": torch.tensor([[0.0, -20.0, 10.0]]),
        "rotation": torch.eye(3).unsqueeze(0),
        "log_focal": torch.tensor([4.0]),
        "intrinsics": torch.tensor(
            [[[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]]
        ),
        "semantic_to_physical": torch.arange(14).unsqueeze(0),
        "raw_pose10d": raw_pose.unsqueeze(0),
    }


def _batch() -> dict[str, object]:
    return {
        "image": torch.zeros(2, 3, 4, 5),
        "targets": {
            "kp": {
                "heatmap": torch.zeros(2, 14, 4, 5),
                "points_xy": torch.zeros(2, 14, 1, 2),
                "point_visible": torch.ones(2, 14, 1, dtype=torch.bool),
                "physical_indices": torch.arange(14)
                .view(1, 14, 1)
                .expand(2, -1, -1)
                .clone(),
            }
        },
        "pose_target": _pose_target(),
        "pose_supervision_mask": torch.tensor([True, False]),
        "image_size": torch.tensor([[4, 5], [4, 5]]),
        "content_size_hw": torch.tensor([[4, 5], [4, 5]]),
    }


@pytest.mark.parametrize("consistency_enabled", [False, True])
def test_pose_objectives_ignore_real_sample_while_dense_loss_uses_it(
    consistency_enabled: bool,
) -> None:
    adapter = MixedCourtPoseModelIOAdapter(
        CourtModelSpec(_bundle(), in_channels=3, short_side=16),
        loss_config=_loss(consistency_enabled=consistency_enabled),
    )
    call = adapter.prepare_training_batch(_batch())
    kp_logits = torch.zeros(2, 14, 4, 5, requires_grad=True)
    predicted_raw = torch.tensor(
        [
            [1.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.2],
            [7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 1.0, 0.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )
    result = adapter.training_result(
        CourtModelOutput({"kp": kp_logits}, CourtRawPoseOutput(predicted_raw)),
        call,
        progress_fraction=1.0,
    )

    changed_real = predicted_raw.detach().clone()
    changed_real[1] = torch.tensor(
        [100.0, -200.0, 300.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    )
    changed_result = adapter.training_result(
        CourtModelOutput(
            {"kp": kp_logits.detach().clone()},
            CourtRawPoseOutput(changed_real),
        ),
        call,
        progress_fraction=1.0,
    )

    torch.testing.assert_close(result.direct_pose_loss, changed_result.direct_pose_loss)
    torch.testing.assert_close(result.loss, changed_result.loss)
    assert (result.consistency is not None) == consistency_enabled
    if result.consistency is not None and changed_result.consistency is not None:
        torch.testing.assert_close(
            result.consistency.auxiliary_loss,
            changed_result.consistency.auxiliary_loss,
        )

    result.loss.backward()
    assert predicted_raw.grad is not None
    assert torch.count_nonzero(predicted_raw.grad[0]).item() > 0
    assert torch.count_nonzero(predicted_raw.grad[1]).item() == 0
    assert kp_logits.grad is not None
    assert torch.count_nonzero(kp_logits.grad[1]).item() > 0


def test_pose_training_rejects_missing_supervision_mask() -> None:
    adapter = MixedCourtPoseModelIOAdapter(
        CourtModelSpec(_bundle(), in_channels=3, short_side=16),
        loss_config=_loss(consistency_enabled=False),
    )
    batch = _batch()
    batch.pop("pose_supervision_mask")

    with pytest.raises(CourtModelIOError, match="requires pose_supervision_mask"):
        adapter.prepare_training_batch(batch)
