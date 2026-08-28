"""Unit coverage for mixed-source Court training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.data.mixed import (
    MixedSourceBatchSampler,
    mixed_court_detection_collate,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtModelOutput,
    CourtModelSpec,
)
from src.tasks.court_detection.model_io.mixed_adapter import (
    MixedCourtPoseModelIOAdapter,
)
from src.tasks.court_detection.models.pose_head import CourtRawPoseOutput
from src.tasks.court_detection.training.runner_mixed import (
    resolve_mixed_training_config,
)

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="kp14",
                output_channels=14,
                channel_names=tuple(f"physical_{index}" for index in range(14)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


def _line_bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "line": CourtTargetSpec(
                kind="line",
                schema="line",
                output_channels=1,
                channel_names=("line",),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


def _loss() -> CourtLossConfig:
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
                "enabled": True,
                "weight": 1.0,
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
    raw_pose = torch.tensor(
        [0.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0]
    )
    return {
        "translation_m": torch.tensor([0.0, -20.0, 10.0]),
        "rotation": torch.eye(3),
        "log_focal": torch.tensor(4.0),
        "intrinsics": torch.tensor(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "semantic_to_physical": torch.arange(14),
        "raw_pose10d": raw_pose,
    }


def _kp_batch() -> dict[str, object]:
    pose = _pose_target()
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
        "pose_target": {
            name: value.unsqueeze(0) for name, value in pose.items()
        },
        "pose_supervision_mask": torch.tensor([True, False]),
        "image_size": torch.tensor([[4, 5], [4, 5]]),
        "content_size_hw": torch.tensor([[4, 5], [4, 5]]),
    }


def test_mixed_sampler_preserves_exact_source_counts_and_cycles() -> None:
    sampler = MixedSourceBatchSampler(
        {"synthetic_court": 3, "tennis_court_detector": 10},
        {"synthetic_court": 2, "tennis_court_detector": 2},
        seed=7,
        shuffle=False,
    )

    batches = list(sampler)

    assert len(batches) == 5
    for batch in batches:
        assert len(batch) == 4
        assert sum(index < 3 for index in batch) == 2
        assert sum(index >= 3 for index in batch) == 2
    assert {index for batch in batches for index in batch if index >= 3} == set(
        range(3, 13)
    )


def test_mixed_collate_stacks_pose_for_synthetic_samples_only() -> None:
    pose = _pose_target()
    synthetic = {
        "image": torch.zeros(3, 4, 5),
        "targets": {"line": torch.zeros(1, 4, 5)},
        "image_size": torch.tensor([4, 5]),
        "content_size_hw": torch.tensor([4, 5]),
        "sample_id": "synthetic",
        "metadata": {"source_kind": "synthetic_court"},
        "pose_target": pose,
    }
    real = {
        "image": torch.zeros(3, 4, 5),
        "targets": {"line": torch.zeros(1, 4, 5)},
        "image_size": torch.tensor([4, 5]),
        "content_size_hw": torch.tensor([4, 5]),
        "sample_id": "real",
        "metadata": {"source_kind": "tennis_court_detector"},
    }

    collated = mixed_court_detection_collate(
        [synthetic, real],
        bundle=_line_bundle(),
    )

    torch.testing.assert_close(
        collated["pose_supervision_mask"],
        torch.tensor([True, False]),
    )
    pose_batch = collated["pose_target"]
    assert isinstance(pose_batch, dict)
    assert pose_batch["translation_m"].shape == (1, 3)
    assert pose_batch["raw_pose10d"].shape == (1, 10)
    assert collated["image"].shape == (2, 3, 8, 8)


def test_pose_and_consistency_ignore_real_sample_pose_output() -> None:
    adapter = MixedCourtPoseModelIOAdapter(
        CourtModelSpec(_bundle(), in_channels=3, short_side=16),
        loss_config=_loss(),
    )
    batch = _kp_batch()
    call = adapter.prepare_training_batch(batch)
    kp_logits = torch.zeros(2, 14, 4, 5, requires_grad=True)
    predicted_raw = torch.tensor(
        [
            [1.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.2],
            [7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 1.0, 0.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )
    result = adapter.training_result(
        CourtModelOutput(
            {"kp": kp_logits},
            CourtRawPoseOutput(predicted_raw),
        ),
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

    assert result.decoded_pose.translation_m.shape == (1, 3)
    assert result.consistency is not None
    assert changed_result.consistency is not None
    torch.testing.assert_close(
        result.direct_pose_loss,
        changed_result.direct_pose_loss,
    )
    torch.testing.assert_close(
        result.consistency.auxiliary_loss,
        changed_result.consistency.auxiliary_loss,
    )
    torch.testing.assert_close(result.loss, changed_result.loss)

    result.loss.backward()
    assert predicted_raw.grad is not None
    assert torch.count_nonzero(predicted_raw.grad[0]).item() > 0
    assert torch.count_nonzero(predicted_raw.grad[1]).item() == 0


def test_train_mixed_config_reuses_both_source_presets() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_mixed")

    standard, mixed = resolve_mixed_training_config(config)

    assert standard.data.source.kind == "synthetic_court"
    assert standard.data.processing.targets[0].kind == "kp"
    assert set(mixed.sources) == {
        "synthetic_court",
        "tennis_court_detector",
    }
    assert dict(mixed.train_batch_counts) == {
        "synthetic_court": 4,
        "tennis_court_detector": 4,
    }


def test_pose_overrides_propagate_to_mixed_synthetic_source() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=[
                "data/augmentation=pose_safe",
                "data.source.keypoint_court_scope=target_court",
                "loss.pose.enabled=true",
                "loss.pose.translation_weight=1.0",
                "loss.pose.rotation_weight=1.0",
                "loss.pose.focal_weight=1.0",
                "loss.consistency.enabled=true",
                "loss.consistency.weight=1.0",
            ],
        )

    _, mixed = resolve_mixed_training_config(config)
    synthetic = mixed.sources["synthetic_court"]

    assert synthetic.kind == "synthetic_court"
    assert synthetic.keypoint_court_scope == "target_court"
