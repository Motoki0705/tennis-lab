"""Contracts for the one-plan Court geometry composition."""

from __future__ import annotations

from types import MappingProxyType
from typing import cast
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.tasks.court_detection.configuration import CourtAugmentationConfig
from src.tasks.court_detection.data.collate import court_detection_collate
from src.tasks.court_detection.data.contracts import (
    CourtKeypointChannels,
    CourtRawSample,
    CourtSampleMetadata,
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.data.processing.geometry import (
    CourtGeometryPlan,
    CourtProcessingGeometry,
)


def _config(**changes: object) -> CourtAugmentationConfig:
    values: dict[str, object] = {
        "train_scales": [32],
        "val_short_side": 32,
        "crop_scale": [0.9, 1.0],
        "crop_ratio": [0.5, 2.0],
        "hflip_prob": 0.0,
        "affine_degrees": 0.0,
        "affine_translate": [0.0, 0.0],
        "affine_scale": [1.0, 1.0],
        "affine_shear": 0.0,
        "perspective_distortion": 0.0,
        "perspective_prob": 0.0,
        "color_jitter": [0.0, 0.0, 0.0, 0.0],
        "gaussian_blur_kernel": [3],
        "gaussian_blur_sigma": [0.1, 0.1],
        "gaussian_blur_prob": 0.0,
        "min_visible_kp": 0,
        "visibility_max_retries": 1,
    }
    values.update(changes)
    return CourtAugmentationConfig.from_mapping(values)


def _raw_sample(
    image_array: np.ndarray,
    *,
    points_xy: torch.Tensor | None = None,
) -> CourtRawSample:
    channels = None
    if points_xy is not None:
        point_count = points_xy.shape[0]
        channels = CourtKeypointChannels(
            channel_names=tuple(f"point-{index}" for index in range(point_count)),
            points_xy=points_xy[:, None],
            point_visible=torch.ones((point_count, 1), dtype=torch.bool),
            physical_indices=torch.arange(point_count)[:, None],
            horizontal_flip_permutation=tuple(range(point_count)),
        )
    return CourtRawSample(
        sample_id="sample",
        image=Image.fromarray(image_array, mode="RGB"),
        keypoint_channels=channels,
        court_instances=(),
        dense_target_refs=MappingProxyType({}),
        metadata=CourtSampleMetadata(
            source_kind="tennis_court_detector",
            source_schema="fixture",
            source_sample_id="sample",
            scene_id=None,
            provenance=MappingProxyType({}),
        ),
    )


def test_train_crop_is_rescaled_to_configured_short_side() -> None:
    geometry = CourtProcessingGeometry(_config(), is_train=True)

    with patch.object(
        geometry,
        "_random_resized_crop",
        return_value=(8, 16, 16, 32),
    ) as crop:
        plan = geometry._sample_once((80, 40))

    crop.assert_called_once_with(32, 64)
    assert plan.output_size_hw == (32, 64)
    torch.testing.assert_close(
        plan.matrix,
        torch.tensor(
            [[1.6, 0.0, -32.0], [0.0, 1.6, -16.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
    )


def test_train_crop_preserves_aspect_ratio_at_the_configured_scale() -> None:
    geometry = CourtProcessingGeometry(_config(), is_train=True)

    with patch.object(
        geometry,
        "_random_resized_crop",
        return_value=(0, 0, 16, 24),
    ):
        plan = geometry._sample_once((80, 40))

    assert plan.output_size_hw == (32, 48)
    assert min(plan.output_size_hw) == 32
    assert plan.output_size_hw[1] / plan.output_size_hw[0] == 1.5


def test_aspect_preserving_train_shapes_are_padded_by_real_collate() -> None:
    geometry = CourtProcessingGeometry(_config(), is_train=True)
    plans = [
        CourtGeometryPlan(
            torch.tensor(
                [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            (32, 64),
            False,
        ),
        CourtGeometryPlan(
            torch.tensor(
                [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            (64, 32),
            False,
        ),
    ]
    raws = [
        _raw_sample(np.zeros((40, 80, 3), dtype=np.uint8)),
        _raw_sample(np.zeros((80, 40, 3), dtype=np.uint8)),
    ]
    samples: list[dict[str, object]] = []
    for raw, plan in zip(raws, plans, strict=True):
        transformed = geometry.apply(raw, dense_targets={}, plan=plan)
        samples.append(
            {
                "sample_id": raw.sample_id,
                "image": transformed.image_tensor,
                "image_size": transformed.image_size,
                "targets": {
                    "line": torch.ones(
                        (1, *plan.output_size_hw), dtype=torch.float32
                    )
                },
                "metadata": transformed.metadata.to_dict(),
            }
        )
    bundle = CourtTargetBundleSpec(
        targets=MappingProxyType(
            {
                "line": CourtTargetSpec(
                    kind="line",
                    schema="fixture_line",
                    output_channels=1,
                    channel_names=("line",),
                    target_dtype=torch.float32,
                    precomputed=True,
                )
            }
        )
    )

    batch = court_detection_collate(samples, bundle=bundle)

    assert cast(torch.Tensor, batch["image"]).shape == (2, 3, 64, 64)
    torch.testing.assert_close(
        cast(torch.Tensor, batch["image_size"]),
        torch.tensor([[32, 64], [64, 32]]),
    )


def test_one_matrix_keeps_rgb_points_and_dense_target_correspondent() -> None:
    source: np.ndarray = np.zeros((12, 20, 3), dtype=np.uint8)
    source[5, 7] = (255, 255, 255)
    raw = _raw_sample(source, points_xy=torch.tensor([[7.0, 5.0]]))
    dense = torch.zeros((12, 20), dtype=torch.uint8)
    dense[5, 7] = 1
    plan = CourtGeometryPlan(
        matrix=torch.tensor(
            [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        output_size_hw=(16, 24),
        horizontal_flipped=False,
    )
    geometry = CourtProcessingGeometry(_config(), is_train=False)

    transformed = geometry.apply(raw, dense_targets={"line": dense}, plan=plan)

    assert transformed.keypoint_channels is not None
    transformed_point = transformed.keypoint_channels.points_xy[0, 0]
    torch.testing.assert_close(transformed_point, torch.tensor([10.0, 7.0]))
    line_peak = torch.nonzero(transformed.dense_targets["line"] == 1)
    assert line_peak.tolist() == [[7, 10]]
    rgb_peak = torch.argmax(transformed.image_tensor.mean(dim=0)).item()
    assert divmod(rgb_peak, 24) == (7, 10)


def test_visibility_retry_returns_earliest_best_candidate() -> None:
    raw = _raw_sample(
        np.zeros((10, 10, 3), dtype=np.uint8),
        points_xy=torch.tensor([[1.0, 1.0], [5.0, 5.0], [8.0, 8.0]]),
    )
    geometry = CourtProcessingGeometry(
        _config(min_visible_kp=3, visibility_max_retries=4),
        is_train=True,
    )
    candidates = [
        CourtGeometryPlan(
            torch.tensor(
                [[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            (10, 10),
            False,
        ),
        CourtGeometryPlan(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            (10, 10),
            False,
        ),
        CourtGeometryPlan(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            (10, 10),
            False,
        ),
        CourtGeometryPlan(
            torch.tensor(
                [[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            (10, 10),
            False,
        ),
    ]

    with patch.object(geometry, "_sample_once", side_effect=candidates) as sampled:
        selected = geometry.sample(raw)

    assert sampled.call_count == 4
    assert selected is candidates[1]


def test_visibility_retry_stops_at_first_candidate_meeting_requirement() -> None:
    raw = _raw_sample(
        np.zeros((10, 10, 3), dtype=np.uint8),
        points_xy=torch.tensor([[1.0, 1.0], [5.0, 5.0], [8.0, 8.0]]),
    )
    geometry = CourtProcessingGeometry(
        _config(min_visible_kp=2, visibility_max_retries=3),
        is_train=True,
    )
    first = CourtGeometryPlan(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        (10, 10),
        False,
    )
    later = CourtGeometryPlan(torch.eye(3, dtype=torch.float64), (10, 10), False)

    with patch.object(geometry, "_sample_once", side_effect=[first, later]) as sampled:
        selected = geometry.sample(raw)

    assert sampled.call_count == 1
    assert selected is first
