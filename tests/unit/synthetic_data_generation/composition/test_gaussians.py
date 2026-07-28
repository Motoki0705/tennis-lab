"""Tests for rigid Gaussian transforms and fail-closed scene concatenation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    compose_gaussians,
    transform_gaussians,
)
from src.synthetic_data_generation.scene_contract import SimilarityTransform

APPEARANCE_SPACE = "a" * 64


def _gaussians(
    *,
    instance_id: int,
    dtype: torch.dtype = torch.float64,
) -> GaussianTensorSet:
    return GaussianTensorSet(
        means=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=dtype,
        ),
        quats=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=dtype,
        ),
        log_scales=torch.log(
            torch.tensor(
                [[1.0, 2.0, 3.0], [0.5, 1.0, 2.0]],
                dtype=dtype,
            )
        ),
        opacity_logits=torch.tensor([0.0, 1.0], dtype=dtype),
        features=torch.arange(8, dtype=dtype).reshape(2, 4),
        instance_ids=torch.full((2,), instance_id, dtype=torch.int64),
        appearance_space_sha256=APPEARANCE_SPACE,
    )


def _rotation_z_90() -> tuple[float, ...]:
    return (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _quat_to_rotation(quaternion: Tensor) -> Tensor:
    normalized = F.normalize(quaternion, dim=-1)
    w, x, y, z = normalized.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _covariances(gaussians: GaussianTensorSet) -> Tensor:
    rotations = _quat_to_rotation(gaussians.quats)
    diagonal = torch.diag_embed(torch.exp(gaussians.log_scales).square())
    return rotations @ diagonal @ rotations.transpose(-1, -2)


def test_similarity_transform_preserves_anisotropic_covariance_geometry() -> None:
    source = _gaussians(instance_id=3)
    transform = SimilarityTransform(
        scale=2.0,
        rotation=_rotation_z_90(),
        translation=(10.0, -1.0, 0.5),
    )

    transformed = transform_gaussians(source, transform)

    torch.testing.assert_close(
        transformed.means,
        torch.tensor(
            [[10.0, 1.0, 0.5], [6.0, -1.0, 0.5]],
            dtype=torch.float64,
        ),
    )
    expected_quaternion = torch.tensor(
        [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        dtype=torch.float64,
    ).expand(2, -1)
    torch.testing.assert_close(transformed.quats, expected_quaternion)
    torch.testing.assert_close(
        transformed.log_scales,
        source.log_scales + math.log(2.0),
    )

    rotation = torch.tensor(_rotation_z_90(), dtype=torch.float64).reshape(3, 3)
    expected_covariances = 4.0 * rotation @ _covariances(source) @ rotation.T
    torch.testing.assert_close(
        _covariances(transformed),
        expected_covariances,
    )
    assert transformed.means.dtype == source.means.dtype
    assert transformed.means.device == source.means.device
    torch.testing.assert_close(transformed.features, source.features)
    torch.testing.assert_close(transformed.opacity_logits, source.opacity_logits)
    torch.testing.assert_close(transformed.instance_ids, source.instance_ids)


def test_compose_concatenates_background_and_unique_instances_in_order() -> None:
    background = _gaussians(instance_id=0, dtype=torch.float32)
    first = _gaussians(instance_id=1, dtype=torch.float32)
    second = _gaussians(instance_id=2, dtype=torch.float32)

    composed = compose_gaussians(background, (first, second))

    assert composed.gaussian_count == 6
    assert composed.feature_dim == 4
    assert composed.instance_ids.tolist() == [0, 0, 1, 1, 2, 2]
    torch.testing.assert_close(composed.means[:2], background.means)
    torch.testing.assert_close(composed.means[2:4], first.means)
    torch.testing.assert_close(composed.means[4:], second.means)


def test_compose_rejects_duplicate_ids_and_independent_appearance() -> None:
    background = _gaussians(instance_id=0)
    first = _gaussians(instance_id=1)

    with pytest.raises(ValueError, match="Duplicate movable instance ids"):
        compose_gaussians(background, (first, _gaussians(instance_id=1)))

    mismatched = replace(first, appearance_space_sha256="b" * 64)
    with pytest.raises(ValueError, match="different NHT appearance space"):
        compose_gaussians(background, (mismatched,))


def test_gaussian_tensor_set_rejects_non_finite_and_mixed_dtype() -> None:
    source = _gaussians(instance_id=1)
    non_finite = source.means.clone()
    non_finite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        replace(source, means=non_finite)

    with pytest.raises(TypeError, match="same dtype"):
        replace(source, features=source.features.float())
