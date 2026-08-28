"""Unit coverage for fixed-ratio mixed Court data contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.data.mixed import (
    MixedSourceBatchSampler,
    _compatible_bundle,
    mixed_court_detection_collate,
)
from src.utils.schema.court import GROUND_COURT_KP_NAMES

pytestmark = pytest.mark.unit


def _kp_bundle(*, schema: str) -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema=schema,
                output_channels=14,
                channel_names=GROUND_COURT_KP_NAMES,
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
                schema="court_line_binary_v1",
                output_channels=1,
                channel_names=("court_line",),
                target_dtype=torch.float32,
                precomputed=True,
            )
        }
    )


def _pose_target() -> dict[str, torch.Tensor]:
    return {
        "translation_m": torch.tensor([0.0, -20.0, 10.0]),
        "rotation": torch.eye(3),
        "log_focal": torch.tensor(4.0),
        "intrinsics": torch.tensor(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "semantic_to_physical": torch.arange(14),
        "raw_pose10d": torch.tensor(
            [0.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0]
        ),
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
    synthetic = {
        "image": torch.zeros(3, 4, 5),
        "targets": {"line": torch.zeros(1, 4, 5)},
        "image_size": torch.tensor([4, 5]),
        "content_size_hw": torch.tensor([4, 5]),
        "sample_id": "synthetic",
        "metadata": {"source_kind": "synthetic_court"},
        "pose_target": _pose_target(),
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
    images = collated["image"]
    assert isinstance(images, torch.Tensor)
    assert images.shape == (2, 3, 8, 8)


def test_mixed_bundle_accepts_only_explicit_semantic_kp_identity_mapping() -> None:
    synthetic = _kp_bundle(
        schema="synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"
    )
    real = _kp_bundle(schema="tennis_court_detector_kp14:gaussian_max_v1")

    assert _compatible_bundle(synthetic, real)

    real_spec = real.targets["kp"]
    reordered = CourtTargetBundleSpec(
        {
            "kp": replace(
                real_spec,
                channel_names=tuple(reversed(real_spec.channel_names)),
            )
        }
    )
    unknown_schema = CourtTargetBundleSpec(
        {"kp": replace(real_spec, schema="unknown_kp14:gaussian_max_v1")}
    )

    assert not _compatible_bundle(synthetic, reordered)
    assert not _compatible_bundle(synthetic, unknown_schema)
